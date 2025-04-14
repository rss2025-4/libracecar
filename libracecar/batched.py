from __future__ import annotations

import functools
import math
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    Sequence,
    TypeVar,
    overload,
)

import equinox as eqx
import jax
from jax import Array, ShapeDtypeStruct, lax
from jax import numpy as jnp
from jax import random
from jax import tree_util as jtu
from jax._src import traceback_util
from jax.util import safe_zip as zip
from jaxtyping import ArrayLike
from typing_extensions import TypeVarTuple

from .utils import (
    blike,
    cast_fsig,
    cast_unchecked,
    ival,
    pp_obj,
    pretty_print,
    shape_of,
)

traceback_util.register_exclusion(__file__)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T1_tup = TypeVarTuple("T1_tup")


def _remove_prefix(
    v: tuple[int, ...], prefix: tuple[int, ...], check: bool = False
) -> tuple[int, ...]:
    if check:
        assert v[: len(prefix)] == prefix
    return v[len(prefix) :]


def _remove_suffix(v: tuple[int, ...], suffix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[len(v) - len(suffix) :] == suffix
    return v[: len(v) - len(suffix)]


def _batched_treemap_of(fn: Callable[Concatenate[tuple[Array, ...], P], Array]):
    @functools.wraps(fn)
    def inner(
        batches: Sequence[batched[T]], *args: P.args, **kwargs: P.kwargs
    ) -> batched[T]:
        return batched_treemap(lambda *bufs: fn(bufs, *args, **kwargs), *batches)

    return inner


def _batched_treemap_of_one(
    fn: Callable[Concatenate[Array, P], Array],
):
    def inner(arg: batched[T], *args: P.args, **kwargs: P.kwargs) -> batched[T]:
        return batched_treemap(lambda buf: fn(buf, *args, **kwargs), arg)

    return inner


class batched(eqx.Module, Generic[T_co]):
    _bufs: list[Array]
    _shapes: list[ShapeDtypeStruct] = eqx.field(static=True)
    _pytree: jtu.PyTreeDef = eqx.field(static=True)
    _tracking: Array | None = None

    @staticmethod
    def create(
        val: T2, batch_dims: tuple[int, ...] = (), *, broadcast: bool = False
    ) -> batched[T2]:
        bufs, tree = jtu.tree_flatten(val)
        if len(bufs) == 0:
            return batched([], [], tree, jnp.zeros(batch_dims))

        try:
            _bufs: list[Array] = []
            _shapes: list[ShapeDtypeStruct] = []

            for x in bufs:
                if not isinstance(x, Array):
                    x = jnp.array(x)
                shape = _remove_prefix(x.shape, batch_dims, check=not broadcast)
                _shapes.append(ShapeDtypeStruct(shape, x.dtype))
                if broadcast:
                    x, _ = jnp.broadcast_arrays(x, jnp.zeros(batch_dims + shape))
                    assert x.shape == batch_dims + shape
                _bufs.append(x)

            return batched(_bufs, _shapes, tree)
        except Exception as e:
            raise Exception(f"failed to create batched: {batch_dims}\n{val}") from e

    def batch_dims(self) -> tuple[int, ...]:
        if self._tracking is not None:
            return self._tracking.shape
        ans = [
            _remove_suffix(b.shape, s.shape) for b, s in zip(self._bufs, self._shapes)
        ]
        for x in ans:
            assert x == ans[0]
        return ans[0]

    @staticmethod
    def _unreduce(bds: tuple[int, ...], val: T2) -> batched[T2]:
        return batched.create(val, bds)

    @staticmethod
    def __reduce__typed(f: Callable[[*T1_tup], T2], obj: tuple["*T1_tup"]):
        return f, obj

    def __reduce__(self):
        return self.__reduce__typed(
            self._unreduce,
            (
                self.batch_dims(),
                self.unflatten(),
            ),
        )

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def count(self) -> int:
        return math.prod(self.batch_dims())

    def unflatten(self) -> T_co:
        return jtu.tree_unflatten(self._pytree, self._bufs)

    @property
    def uf(self):
        return self.unflatten()

    def unwrap(self) -> T_co:
        assert self.batch_dims() == ()
        return self.unflatten()

    def __repr__(self):
        try:
            bds = self.batch_dims()
            return pp_obj("batched", repr(bds), pretty_print(self.unflatten())).format()
        except:
            return pp_obj("malformed_batched", pretty_print(self.unflatten())).format()

    @staticmethod
    @cast_fsig(jnp.arange)
    def arange(*args, **kwargs) -> batched[Array]:
        ans = jnp.arange(*args, **kwargs)
        (n,) = ans.shape
        return batched.create(ans, (n,))

    def repeat(self, n: int) -> batched[T_co]:
        return jax.vmap(lambda: self, axis_size=n)()

    def reshape(self, *new_shape: int) -> batched[T_co]:
        bd = self.batch_dims()

        if new_shape == (-1,) and len(bd) == 1:
            return self

        return jtu.tree_map(
            lambda x: x.reshape(*new_shape, *x.shape[len(bd) :]),
            self,
        )

    concat = staticmethod(_batched_treemap_of(jnp.concat))
    stack = staticmethod(_batched_treemap_of(jnp.stack))
    roll = _batched_treemap_of_one(jnp.roll)
    mean = _batched_treemap_of_one(jnp.mean)
    sum = _batched_treemap_of_one(jnp.sum)

    def __getitem__(self, idx: Any) -> batched[T_co]:
        return batched_treemap(lambda x: x[idx], self)

    def dynamic_slice(
        self, start_indices: Sequence[ArrayLike], slice_sizes: Sequence[int]
    ) -> batched[T_co]:
        bds = self.batch_dims()
        assert len(bds) == len(start_indices)
        assert len(bds) == len(slice_sizes)

        def slice_one(x):
            s = shape_of(x)
            rest = _remove_prefix(s, bds)
            return lax.dynamic_slice(
                x, [*start_indices, *(0 for _ in rest)], [*slice_sizes, *rest]
            )

        return jtu.tree_map(slice_one, self)

    def __len__(self) -> int:
        return self.batch_dims()[0]

    def map(self, f: Callable[[T_co], T2], /, *, sequential=False) -> batched[T2]:
        return batched_vmap(f, self, sequential=sequential)

    def map_with_rng(
        self, rng_key: ArrayLike, f: Callable[[T_co, Array], T2], /, *, sequential=False
    ) -> batched[T2]:
        bds = self.batch_dims()
        keys = batched.create(random.split(rng_key, bds), bds)
        return batched_vmap(f, self, keys, sequential=sequential)

    def tuple_map(
        self: batched[tuple["*T1_tup"]], f: Callable[[*T1_tup], T2]
    ) -> batched[T2]:
        return batched_vmap(lambda x: f(*x), self)

    @overload
    def split_tuple(
        self: batched[tuple[T1, T2]],
    ) -> tuple[batched[T1], batched[T2]]: ...

    @overload
    def split_tuple(
        self: batched[tuple[T1, T2, T3]],
    ) -> tuple[batched[T1], batched[T2], batched[T3]]: ...

    def split_tuple(self: batched[tuple]) -> tuple[batched, ...]:
        bds = self.batch_dims()
        me = self.unflatten()
        assert isinstance(me, tuple)
        return tuple(batched.create(x, bds) for x in me)

    def static_map(self, f: Callable[[T_co], T2], /) -> T2:
        if self.batch_dims() == ():
            return f(self.unwrap())

        def inner(x: batched[T_co]):
            return x.static_map(f)

        return jax.vmap(inner, out_axes=None)(self)

    def filter(self, f: Callable[[T_co], blike]) -> tuple[batched[T_co], ival]:
        return self.filter_arr(self.map(f))

    def filter_arr(self, bools_: batched[blike]) -> tuple[batched[T_co], ival]:
        (n,) = self.batch_dims()

        bools = bools_.unflatten()
        assert isinstance(bools, Array)
        assert bools.shape == (n,)

        idxs = jnp.nonzero(bools, size=n, fill_value=0)

        ans: batched[T_co] = jtu.tree_map(lambda x: x[idxs], self)
        assert ans.batch_dims() == (n,)
        return ans, bools.sum()

    def scan(
        self, f: Callable[[T2, T_co], tuple[T2, T3]], init: T2
    ) -> tuple[T2, batched[T3]]:
        (_,) = self.batch_dims()

        def inner(c: T2, x: batched[T_co]):
            c, y = f(c, x.unwrap())
            return c, batched.create(y)

        return lax.scan(inner, init=init, xs=self)

    def thread(self: batched[Callable[[T2], T2]], init: T2) -> T2:
        (_,) = self.batch_dims()

        def inner(c: T2, f: batched[Callable[[T2], T2]]):
            return f.unwrap()(c), None

        ans, _ = lax.scan(inner, init=init, xs=self)
        return ans

    def all_idxs(self) -> batched[tuple[ival, ...]]:
        bds = self.batch_dims()
        if len(bds) == 0:
            return batched.create(())
        n = bds[0]

        def inner(i: batched[ival], v: batched[T_co]) -> batched[tuple[ival, ...]]:
            return v.all_idxs().map(lambda rest: (i.unwrap(), *rest))

        return jax.vmap(inner)(batched.arange(n), self)

    def enumerate(
        self,
        f: (
            Callable[[T_co], R]
            | Callable[[T_co, ival], R]
            | Callable[[T_co, ival, ival], R]
            | Callable[[T_co, ival, ival, ival], R]
        ),
    ) -> batched[R]:
        f_ = cast_unchecked["Callable[[T_co, *tuple[ival, ...]], R]"]()(f)
        idxs = self.all_idxs()
        return batched_zip(self, idxs).tuple_map(lambda v, idx: f_(v, *idx))

    def split_batch_dims(
        self,
        *,
        outer: tuple[int, ...] | None = None,
        inner: tuple[int, ...] | None = None,
    ) -> batched[batched[T_co]]:
        assert not inner is None and outer is None
        bds = self.batch_dims()

        if inner is None:
            inner = bds[len(outer) :]
        if outer is None:
            outer = bds[: -len(inner)]

        assert bds == outer + inner
        return batched.create(self, outer)

    def sort(self, key: Callable[[T_co], Array]):
        def inner(v: T_co):
            ans = key(v)
            assert isinstance(ans, Array)
            assert ans.shape == ()
            return ans

        keys = self.map(inner)
        sorted_indices = jnp.argsort(keys.uf)
        return self[sorted_indices]


@overload
def batched_vmap(
    f: Callable[[T1], R], a1: batched[T1], /, *, sequential=False
) -> batched[R]: ...
@overload
def batched_vmap(
    f: Callable[[T1, T2], R], a1: batched[T1], a2: batched[T2], /, *, sequential=False
) -> batched[R]: ...
@overload
def batched_vmap(
    f: Callable[[T1, T2, T3], R],
    a1: batched[T1],
    a2: batched[T2],
    a3: batched[T3],
    /,
    *,
    sequential=False,
) -> batched[R]: ...


def batched_vmap(f: Callable[..., R], *args: batched, sequential=False) -> batched[R]:
    bds = [x.batch_dims() for x in args]
    for bd in bds:
        assert bd == bds[0]
    if bds[0] == ():
        return batched.create(f(*(x.unwrap() for x in args)))

    def inner(*args: batched) -> batched[R]:
        return batched_vmap(f, *args, sequential=sequential)

    if not sequential:
        return jax.vmap(inner)(*args)
    else:
        _, ans = lax.scan(lambda _, args: (None, inner(*args)), init=None, xs=args)
        return ans


def batched_zip(a1: batched[T1], a2: batched[T2], /) -> batched[tuple[T1, T2]]:
    return batched_vmap(lambda *args: args, a1, a2)


@overload
def batched_treemap(f: Callable[[Array], Array], a1: batched[T], /) -> batched[T]: ...
@overload
def batched_treemap(
    f: Callable[[Array, Array], Array], a1: batched[T], a2: batched[T], /
) -> batched[T]: ...
@overload
def batched_treemap(
    f: Callable[[*tuple[Array, ...]], Array], /, *args: batched[T]
) -> batched[T]: ...


def batched_treemap(f: Callable[..., Array], /, *args: batched[T]) -> batched[T]:
    try:
        for x in args:
            assert x._pytree == args[0]._pytree
            assert x._shapes == args[0]._shapes
            for s1, s2 in zip(x._shapes, args[0]._shapes):
                assert s1.shape == s2.shape
    except Exception as e:
        raise TypeError("invalid args:", args) from e

    # bds = [x.batch_dims() for x in args]

    def handle_one(shape: tuple[int, ...], *bufs: Array):
        if len(shape) == 0:
            return f(*bufs)
        else:
            return jax.vmap(
                lambda *bufs: handle_one(shape[:-1], *bufs),
                in_axes=-1,
                out_axes=-1,
                axis_size=shape[-1],
            )(*bufs)

    new_bufs = [
        handle_one(s.shape, *bufs)
        for (s, *bufs) in zip(args[0]._shapes, *(x._bufs for x in args))
    ]
    # ShapeDtypeStruct(_remove_prefix(x.shape, batch_dims), x.dtype)

    ans = batched(
        _bufs=new_bufs,
        _shapes=[
            ShapeDtypeStruct(x.shape, dtype=b.dtype)
            for x, b in zip(args[0]._shapes, new_bufs)
        ],
        _pytree=args[0]._pytree,
    )
    _ = ans.batch_dims()
    return ans
