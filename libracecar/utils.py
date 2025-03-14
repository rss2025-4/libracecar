from threading import Thread
from typing import Annotated, Any, Callable, Generic, ParamSpec, TypeVar, final

import equinox as eqx
import jax
import jax._src.pretty_printer as pp
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.core import Tracer, eval_jaxpr
from jax.experimental import io_callback
from jaxtyping import Array, ArrayLike, Bool, Float, Int

F = TypeVar("F", bound=Callable)
T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


fval = Float[Array, ""]
ival = Int[Array, ""]
bval = Bool[Array, ""]

fpair = Float[Array, "2"]

flike = fval | float

# type annoation for a batched pytree
batched = Annotated


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


# jit = eqx.filter_jit
def jit(f: F) -> F:
    return cast_unchecked()(jax.jit(f))


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
    if isinstance(x, Tracer):
        return x._pretty_print()
    return pp_join(*str(x).splitlines())


def _pp_doc(x: pp.Doc | str) -> pp.Doc:
    if isinstance(x, pp.Doc):
        return x
    return pp.text(x)


def pp_join(*docs: pp.Doc | str, sep: pp.Doc | str = pp.brk()) -> pp.Doc:
    return pp.join(_pp_doc(sep), [_pp_doc(x) for x in docs])


def pp_nested(*docs: pp.Doc | str) -> pp.Doc:
    return pp.group(pp.nest(2, pp_join(*docs)))


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


def flatten_n_tree(obj: batched[T, ...], n: int) -> batched[T, 0]:
    leaves: list[Array]
    leaves, treedef = jtu.tree_flatten(obj)
    flat_dims = leaves[0].shape[:n]
    for x in leaves:
        if x.shape[:n] != flat_dims:
            raise TypeError(f"flatten_n_tree {n}:\n{obj}")
    if n == 1:
        return obj
    leaves = [jnp.reshape(x, (-1, *x.shape[n:])) for x in leaves]
    return jtu.tree_unflatten(treedef, leaves)


def tree_stack(*objs: T) -> batched[T, 0]:
    return jtu.tree_map(lambda *xs: jnp.stack(xs), *objs)


def _check_shape(a: tuple[int, ...], b: tuple[int, ...]):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y and x != -1 and y != -1:
            return False
    return True


def check_shape(x, *shape: int):
    if not _check_shape(x.shape, shape):
        raise TypeError(f"expected shape {shape}, got\n{x}")


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
