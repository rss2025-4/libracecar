from __future__ import annotations

import math
from typing import (
    Annotated,
    Any,
    Callable,
    Concatenate,
    Generic,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    final,
    overload,
)

import equinox as eqx
import jax
from jax import Array, ShapeDtypeStruct, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.util import safe_zip as zip
from jaxtyping import ArrayLike
from typing_extensions import TypeVarTuple

from .utils import blike, ival, pp_obj, pretty_print, shape_of, tree_at_

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T1_tup = TypeVarTuple("T1_tup")


def _remove_prefix(v: tuple[int, ...], prefix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[: len(prefix)] == prefix
    return v[len(prefix) :]


def _remove_suffix(v: tuple[int, ...], suffix: tuple[int, ...]) -> tuple[int, ...]:
    assert v[len(v) - len(suffix) :] == suffix
    return v[: len(v) - len(suffix)]


def _batched_treemap_of(fn: Callable[Concatenate[tuple[Array, ...], P], Array]):
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

    @staticmethod
    def create(val: T2, batch_dims: tuple[int, ...] = ()) -> batched[T2]:
        bufs, tree = jtu.tree_flatten(val)
        assert len(bufs) > 0
        _bufs: list[Array] = []
        _shapes: list[ShapeDtypeStruct] = []

        for x in bufs:
            if not isinstance(x, Array):
                x = jnp.array(x)
            _bufs.append(x)
            _shapes.append(
                ShapeDtypeStruct(_remove_prefix(x.shape, batch_dims), x.dtype)
            )
        return batched(_bufs, _shapes, tree)

    def batch_dims(self) -> tuple[int, ...]:
        ans = [
            _remove_suffix(b.shape, s.shape) for b, s in zip(self._bufs, self._shapes)
        ]
        for x in ans:
            assert x == ans[0]
        return ans[0]

    def count(self) -> int:
        return math.prod(self.batch_dims())

    def unflatten(self) -> T_co:
        return jtu.tree_unflatten(self._pytree, self._bufs)

    def unwrap(self) -> T_co:
        assert self.batch_dims() == ()
        return self.unflatten()

    def __repr__(self):
        try:
            bds = self.batch_dims()
            return pp_obj("batched", repr(bds), pretty_print(self.unflatten())).format()
        except:
            return pp_obj("malformed_batched", pretty_print(self.unflatten())).format()

    def repeat(self, n: int) -> batched[T_co]:
        return jax.vmap(lambda: self, axis_size=n)()

    def reshape(self, *new_shape: int) -> batched[T_co]:
        bd = self.batch_dims()

        return jtu.tree_map(
            lambda x: x.reshape(*new_shape, *x.shape[len(bd) :]),
            self,
        )

    concat = staticmethod(_batched_treemap_of(jnp.concat))
    stack = staticmethod(_batched_treemap_of(jnp.stack))
    roll = _batched_treemap_of_one(jnp.roll)

    def __getitem__(self, idx: Any) -> batched[T_co]:
        ans = jtu.tree_map(lambda x: x[idx], self)
        _ = ans.batch_dims()
        return ans

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

    def map(self, f: Callable[[T_co], T2]) -> batched[T2]:
        return batched_vmap(f, self)

    def tuple_map(
        self: "batched[tuple[*T1_tup]]", f: Callable[[*T1_tup], T2]
    ) -> batched[T2]:
        return batched_vmap(lambda x: f(*x), self)

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


@overload
def batched_vmap(f: Callable[[T1], R], a1: batched[T1], /) -> batched[R]: ...
@overload
def batched_vmap(
    f: Callable[[T1, T2], R], a1: batched[T1], a2: batched[T2], /
) -> batched[R]: ...
@overload
def batched_vmap(
    f: Callable[[T1, T2, T3], R], a1: batched[T1], a2: batched[T2], a3: batched[T3], /
) -> batched[R]: ...


def batched_vmap(f: Callable[..., R], *args: batched) -> batched[R]:
    bds = [x.batch_dims() for x in args]
    for bd in bds:
        assert bd == bds[0]
    if bds[0] == ():
        return batched.create(f(*(x.unwrap() for x in args)))

    def inner(*args: batched) -> batched[R]:
        return batched_vmap(f, *args)

    return jax.vmap(inner)(*args)


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
