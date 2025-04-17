import functools
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
    overload,
)

import equinox as eqx
import numpy as np
from geometry_msgs.msg import Point
from jax import Array, lax
from jax import numpy as jnp
from std_msgs.msg import ColorRGBA
from termcolor import colored
from visualization_msgs.msg import Marker

from .batched import batched, batched_treemap
from .utils import (
    cast_unchecked,
    cast_unchecked_,
    debug_callback,
    flike,
    fpair,
    fval,
    ival,
    pformat_repr,
)

T = TypeVar("T")
R = TypeVar("R", covariant=True)
P = ParamSpec("P")


class plot_style(eqx.Module):
    color: tuple[fval, fval, fval]
    alpha: fval

    def __init__(
        self, *, color: tuple[flike, flike, flike] = (0.0, 0.0, 0.0), alpha: flike = 1.0
    ):
        self.color = cast_unchecked_(
            tuple(jnp.array(x, dtype=np.float32) for x in color)
        )
        self.alpha = jnp.array(alpha, dtype=np.float32)

    def clip(self) -> "plot_style":
        clamp_f = lambda x: lax.min(lax.max(x, 0.001), 0.999)
        r, g, b = self.color
        a = self.alpha
        return plot_style(color=(clamp_f(r), clamp_f(g), clamp_f(b)), alpha=clamp_f(a))


class plot_point(eqx.Module):
    x: fval
    y: fval

    style: plot_style

    @staticmethod
    def create(
        loc: tuple[flike, flike] | fpair, style: plot_style = plot_style()
    ) -> batched["plot_point"]:
        x, y = loc
        return batched.create(
            plot_point(
                x=jnp.array(x, dtype=np.float32),
                y=jnp.array(y, dtype=np.float32),
                style=style,
            )
        )

    def clip(self):
        return plot_point(self.x, self.y, self.style.clip())

    def __call__(self, ctx: "plot_ctx"):
        return ctx.point_batched(batched.create(self))

    __repr__ = pformat_repr


class plot_ctx(eqx.Module):
    idx: ival
    limit: int = eqx.field(static=True)
    points: batched[plot_point]

    @staticmethod
    def create(limit: int) -> "plot_ctx":
        points_buf = batched.create(
            plot_point(jnp.array(0.0), jnp.array(0.0), plot_style())
        ).repeat(limit)
        return plot_ctx(
            idx=jnp.array(0),
            limit=limit,
            points=points_buf,
        )

    def point_batched(self, p: batched[plot_point]) -> "plot_ctx":
        p = p.reshape(-1)

        def update_one(x: Array, y: Array):
            return lax.dynamic_update_slice(
                x, y, (self.idx,), allow_negative_indices=False
            )

        return plot_ctx(
            idx=self.idx + len(p),
            limit=self.limit,
            points=batched_treemap(update_one, self.points, p),
        )

    def __add__(self, f: "plotable") -> "plot_ctx":
        assert isinstance(f, batched)
        while isinstance((uf := f.unflatten()), batched):
            f = uf

        f_ = cast_unchecked[batched[Callable[[plot_ctx], plot_ctx]]]()(f)

        if isinstance(uf, plot_point):
            return self.point_batched(cast_unchecked_(f))
        ctx = f_.reshape(-1).thread(self)
        return ctx

    def _do_warn(self):
        debug_callback(
            lambda idx: print(
                colored(
                    f"too many points: limit is {self.limit}, trying to draw {idx}",
                    "red",
                )
            ),
            self.idx,
        )

    def check(self):
        lax.cond(
            self.idx < self.limit,
            true_fun=lambda: None,
            false_fun=lambda: self._do_warn(),
        )

    def execute(self, m: Marker):
        idx = int(self.idx)

        def to_python(x) -> list[float]:
            ans = np.array(x).tolist()
            assert isinstance(ans, list)
            return ans  # type: ignore

        _points = self.points[:idx].unflatten()
        X = to_python(_points.x)
        Y = to_python(_points.y)

        R, G, B = map(to_python, _points.style.color)
        A = to_python(_points.style.alpha)

        assert isinstance(m.points, list)
        assert isinstance(m.colors, list)

        for x, y, r, g, b, a in zip(X, Y, R, G, B, A):
            p = Point()
            p.x = x
            p.y = y
            c = ColorRGBA()
            c.r, c.g, c.b = r, g, b
            c.a = a

            m.points.append(p)
            m.colors.append(c)


plotable = batched[Callable[[plot_ctx], plot_ctx]] | batched["plotable"]


class plotfn_(eqx.Module):
    f: Callable = eqx.field(static=True)
    args: Any
    kwargs: Any

    def __call__(self, ctx: plot_ctx):
        return self.f(ctx, *self.args, **self.kwargs)


def plotfn(f: Callable[Concatenate[plot_ctx, P], plot_ctx]) -> Callable[P, plotable]:

    @functools.wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> plotable:
        return batched.create(plotfn_(f, args, kwargs))

    return inner


class plotmethod_(eqx.Module):
    f: Callable = eqx.field(static=True)
    self_: Any
    args: Any
    kwargs: Any

    def __call__(self, ctx: plot_ctx):
        return self.f(self.self_, ctx, *self.args, **self.kwargs)


@dataclass
class plotmethod(Generic[T, P]):

    f: Callable[Concatenate[T, plot_ctx, P], plot_ctx]

    def __call__(self, obj: T, /, *args: P.args, **kwargs: P.kwargs) -> plotable:
        return batched.create(plotmethod_(self.f, obj, args, kwargs))

    @overload
    def __get__(
        self, instance: None, owner: type | None = None, /
    ) -> "plotmethod[T, P]": ...

    @overload
    def __get__(
        self, instance: T, owner: type | None = None, /
    ) -> Callable[P, plotable]: ...

    def __get__(self, instance: T | None, owner: type | None = None) -> Any:
        if instance is None:
            return self
        else:

            def inner(*args: P.args, **kwargs: P.kwargs):
                return batched.create(plotmethod_(self.f, instance, args, kwargs))

            return inner
