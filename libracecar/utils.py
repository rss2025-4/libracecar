import dataclasses
import time
from dataclasses import dataclass
from threading import Thread
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    TypeVar,
    final,
    overload,
)

import equinox as eqx
import jax
import jax._src.pretty_printer as pp
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.core import Tracer, eval_jaxpr
from jax.experimental import io_callback
from jaxtyping import Array, ArrayLike, Bool, Complex64, Float, Int, Int32

F = TypeVar("F", bound=Callable)
T = TypeVar("T")
T2 = TypeVar("T2")
N = TypeVar("N")
R = TypeVar("R", covariant=True)
P = ParamSpec("P")
R2 = TypeVar("R2", covariant=True)
P2 = ParamSpec("P2")
JitP = ParamSpec("JitP")


fval = Float[Array, ""]
ival = Int[Array, ""]
bval = Bool[Array, ""]
cval = Complex64[Array, ""]

fpair = Float[Array, "2"]

flike = Float[ArrayLike, ""]
blike = Bool[ArrayLike, ""]
ilike = Bool[Int32, ""]


@final
class _empty_t:
    pass


_empty = _empty_t()


class cast(Generic[T]):
    def __init__(self, _: T | _empty_t = _empty) -> None:
        pass

    def __call__(self, a: T) -> T:
        return a


class cast_unchecked(Generic[T]):
    def __init__(self, _: T | _empty_t = _empty) -> None:
        pass

    def __call__(self, a) -> T:
        return a


cast_unchecked_ = cast_unchecked()


def cast_fsig(f1: Callable[P, Any]):
    def inner(f2: Callable[..., R]) -> Callable[P, R]:
        return f2

    return inner


# jit = eqx.filter_jit
class _jit_wrapped(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    @overload
    def __get__(
        self: "_jit_wrapped[Concatenate[T, P2], R2]",
        obj: T,
        objtype: type | None = None,
        /,
    ) -> Callable[P2, R2]: ...
    @overload
    def __get__(
        self, obj: None, objtype: type | None = None, /
    ) -> "_jit_wrapped[P, R]": ...
    def trace(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Traced: ...
    def lower(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Lowered: ...


class _jit_fn(Protocol[JitP]):
    def __call__(
        self, f: Callable[P, R], /, *args: JitP.args, **kwargs: JitP.kwargs
    ) -> _jit_wrapped[P, R]: ...


def _wrap_jit(
    jit_fn: Callable[Concatenate[Callable, P], Any],
) -> _jit_fn[P]:
    return jit_fn


jit = _wrap_jit(jax.jit)


def io_callback_(
    fn: Callable[P, R], result_shape_dtypes: R = None, ordered: bool = True
):
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        return io_callback(fn, result_shape_dtypes, *args, ordered=ordered, **kwargs)

    return inner


def _callback_wrapped(fn: Callable, tree: jtu.PyTreeDef, np_printoptions):
    def inner(*bufs):
        args, kwargs = jtu.tree_unflatten(tree, bufs)
        with np.printoptions(**np_printoptions):
            fn(*args, **kwargs)

    return inner


def debug_callback(fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs):
    bufs, tree = jtu.tree_flatten((args, kwargs))
    jax.debug.callback(
        _callback_wrapped(fn, tree, np.get_printoptions()), *bufs, ordered=True
    )


@cast(print)
def debug_print(*args, **kwargs):
    debug_callback(print, *args, **kwargs)


def pretty_print(x: Any) -> pp.Doc:
    if isinstance(x, pp.Doc):
        return x
    if isinstance(x, Tracer):
        return x._pretty_print()
    return pp_join(*repr(x).splitlines())


def _pp_doc(x: pp.Doc | str) -> pp.Doc:
    if isinstance(x, pp.Doc):
        return x
    return pp.text(x)


def pp_join(*docs: pp.Doc | str, sep: pp.Doc | str = pp.brk()) -> pp.Doc:
    return pp.join(_pp_doc(sep), [_pp_doc(x) for x in docs])


def pp_nested(*docs: pp.Doc | str) -> pp.Doc:
    return pp.group(pp.nest(2, pp_join(*docs)))


# modified from equinox
_comma_sep = pp.concat([pp.text(","), pp.brk()])


def bracketed(
    name: Optional[pp.Doc],
    indent: int,
    objs: Sequence[pp.Doc],
    lbracket: str,
    rbracket: str,
) -> pp.Doc:
    nested = pp.concat(
        [
            pp.nest(indent, pp.concat([pp.brk(""), pp.join(_comma_sep, objs)])),
            pp.brk(""),
        ]
    )
    concated = []
    if name is not None:
        concated.append(name)
    concated.extend([pp.text(lbracket), nested, pp.text(rbracket)])
    return pp.group(pp.concat(concated))


def named_objs(pairs):
    return [
        pp.concat([pp.text(key + "="), pretty_print(value)]) for key, value in pairs
    ]


def pformat_dataclass(obj) -> pp.Doc:
    objs = named_objs(
        [
            (field.name, getattr(obj, field.name, pp.text("<uninitialised>")))
            for field in dataclasses.fields(obj)
            if field.repr
        ]
    )
    return bracketed(
        name=pp.text(obj.__class__.__name__),
        indent=2,
        objs=objs,
        lbracket="(",
        rbracket=")",
    )


def pformat_repr(self: Any):
    return pformat_dataclass(self).format()


def pp_obj(name: pp.Doc | str, *fields: pp.Doc | str):
    # modified from equinox._pretty_print
    _comma_sep = pp.concat([pp.text(","), pp.brk()])
    nested = pp.concat(
        [
            pp.nest(
                2,
                pp.concat(
                    [pp.brk(""), pp.join(_comma_sep, [_pp_doc(x) for x in fields])]
                ),
            ),
            pp.brk(""),
        ]
    )
    return pp.group(pp.concat([_pp_doc(name), pp.text("("), nested, pp.text(")")]))


# def _check_shape(a: tuple[int, ...], b: tuple[int, ...]):
#     if len(a) != len(b):
#         return False
#     for x, y in zip(a, b):
#         if x != y and x != -1 and y != -1:
#             return False
#     return True


# def check_shape(x, *shape: int):
#     if not _check_shape(x.shape, shape):
#         raise TypeError(f"expected shape {shape}, got\n{x}")


def safe_select(
    pred: ArrayLike,
    on_true: Callable[[], T],
    on_false: Callable[[], T],
) -> T:
    # https://github.com/jax-ml/jax/issues/1052

    def handle_side(assume_side: bool, f: Callable[[], T]):
        jaxpr, shapes = jax.make_jaxpr(f, return_shape=True)()
        consts = [
            lax.select(
                pred,
                on_true=x if assume_side else lax.stop_gradient(x),
                on_false=lax.stop_gradient(x) if assume_side else x,
            )
            for x in jaxpr.consts
        ]
        out_bufs = eval_jaxpr(jaxpr.jaxpr, consts)
        return jtu.tree_unflatten(jtu.tree_structure(shapes), out_bufs)

    return tree_select(
        pred,
        on_true=handle_side(True, on_true),
        on_false=handle_side(False, on_false),
    )


def tree_select(
    pred: ArrayLike,
    on_true: T,
    on_false: T,
) -> T:
    pred_ = jnp.array(pred)
    bufs_true, tree1 = jtu.tree_flatten(on_true)
    bufs_false, tree2 = jtu.tree_flatten(on_false)
    assert tree1 == tree2
    out_bufs = [
        x if x is y else lax.select(pred_, on_true=x, on_false=y)
        for x, y in zip(bufs_true, bufs_false)
    ]
    return jtu.tree_unflatten(tree1, out_bufs)


class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)  # type: ignore
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def shape_of(x: ArrayLike) -> tuple[int, ...]:
    return jnp.shape(x)


def tree_at_(
    where: Callable[[T], N],
    pytree: T,
    replace: N | None = None,
    replace_fn: Callable[[N], N] | None = None,
) -> T:
    kwargs = {}
    if replace is not None:
        kwargs["replace"] = replace
    if replace_fn is not None:
        kwargs["replace_fn"] = replace_fn

    return eqx.tree_at(where=where, pytree=pytree, **kwargs)


def cond_(pre: blike, true_fun: Callable[[], T], false_fun: Callable[[], T]) -> T:
    return lax.cond(pre, true_fun=true_fun, false_fun=false_fun)


def round_clip(x: flike, min: ilike, max: ilike):
    return jnp.clip(jnp.round(x).astype(np.int32), min=min, max=max - 1)


def _ensure_not_weak_typed(x: ArrayLike):
    if isinstance(x, Array) and not x.weak_type:
        return x
    if not isinstance(x, Array):
        x = jnp.array(x)
    return jnp.array(x, dtype=x.dtype)


def ensure_not_weak_typed(x: T) -> T:
    bufs, treedef = jtu.tree_flatten(x)
    return jtu.tree_unflatten(
        treedef,
        [_ensure_not_weak_typed(b) for b in bufs],
    )


def tree_to_ShapeDtypeStruct(v: T) -> T:
    def inner(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return x
        x = jnp.array(x)
        return jax.ShapeDtypeStruct(x.shape, x.dtype)

    return jtu.tree_map(inner, v)


@dataclass
class timer:
    _time: float | None = None
    _time_fn: Callable[[], float] = time.time

    @staticmethod
    def create(fn: Callable[[], float] = time.time):
        return timer(fn(), fn)

    def update(self) -> float:
        assert self._time is not None
        new_t = self._time_fn()
        assert new_t >= self._time
        ans = new_t - self._time
        self._time = new_t
        return ans

    def __enter__(self):
        self.update()
        return self

    @property
    def val(self):
        assert self._time is not None
        ans = self._time_fn() - self._time
        assert ans >= 0
        return ans

    def __exit__(self, exc_type, exc_value, traceback):
        self._time = None


class lazy(eqx.Module, Generic[T]):
    fn: Callable[..., T] = eqx.field(static=True)
    args: Any
    kwargs: Any

    def __init__(self, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)

    @staticmethod
    def get(v: "lazylike[T2]") -> T2:
        if isinstance(v, lazy):
            return v()
        return v


lazylike = lazy[T] | T
