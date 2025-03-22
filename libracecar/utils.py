import dataclasses
import multiprocessing
import os
import traceback
from threading import Thread
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
)

import equinox as eqx
import jax
import jax._src.pretty_printer as pp
import numpy as np
import numpyro
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.core import Tracer, eval_jaxpr
from jax.experimental import io_callback
from jaxtyping import Array, ArrayLike, Bool, Float, Int
from numpyro.distributions import constraints

F = TypeVar("F", bound=Callable)
T = TypeVar("T")
N = TypeVar("N")
R = TypeVar("R", covariant=True)
P = ParamSpec("P")
R2 = TypeVar("R2", covariant=True)
P2 = ParamSpec("P2")
JitP = ParamSpec("JitP")


fval = Float[Array, ""]
ival = Int[Array, ""]
bval = Bool[Array, ""]

fpair = Float[Array, "2"]

flike = Float[ArrayLike, ""]
blike = Bool[ArrayLike, ""]


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


# jit = eqx.filter_jit
class _jit_wrapped(Generic[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def __get__(
        self: "_jit_wrapped[Concatenate[T, P2], R2]", obj: T, *_, **__
    ) -> Callable[P2, R2]: ...
    def trace(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Traced: ...
    def lower(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Lowered: ...


class _jit_fn(Generic[JitP]):
    def __call__(
        self, f: Callable[P, R], /, *args: JitP.args, **kwargs: JitP.kwargs
    ) -> _jit_wrapped[P, R]: ...


def _wrap_jit(
    jit_fn: Callable[Concatenate[Callable, P], Any],
) -> _jit_fn[P]:
    return jit_fn  # type: ignore


jit = _wrap_jit(jax.jit)


def io_callback_(fn: Callable[P, R], result_shape_dtypes: R = None):
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        return io_callback(
            fn, result_shape_dtypes=result_shape_dtypes, ordered=True, *args, **kwargs
        )

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
    on_true: Callable[[], ArrayLike],
    on_false: Callable[[], ArrayLike],
) -> ArrayLike:
    # https://github.com/jax-ml/jax/issues/1052

    def handle_side(assume_side: bool, f: Callable[[], ArrayLike]):
        jaxpr = jax.make_jaxpr(f)()
        consts = [
            lax.select(
                pred,
                on_true=x if assume_side else lax.stop_gradient(x),
                on_false=lax.stop_gradient(x) if assume_side else x,
            )
            for x in jaxpr.consts
        ]
        (ans,) = eval_jaxpr(jaxpr.jaxpr, consts)
        return ans

    return lax.select(
        pred,
        on_true=handle_side(True, on_true),
        on_false=handle_side(False, on_false),
    )


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


def numpyro_param(
    name: str,
    init_value: ArrayLike,
    constraint: constraints.Constraint = constraints.real,
) -> Array:
    ans = numpyro.param(name, init_value, constraint=constraint)
    assert ans is not None
    return jnp.array(ans)


def shape_of(x: ArrayLike) -> tuple[int, ...]:
    return core.get_aval(x).shape  # type: ignore


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
