import math
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
)

import equinox as eqx
import jax
import numpy as np
from beartype import beartype as typechecker
from geometry_msgs.msg import Point
from jax import Array, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jaxtyping import jaxtyped
from std_msgs.msg import ColorRGBA, Float32
from termcolor import colored
from visualization_msgs.msg import Marker

from .batched import batched, batched_treemap
from .utils import (
    cast,
    cast_unchecked,
    cast_unchecked_,
    debug_callback,
    debug_print,
    flike,
    fpair,
    fval,
    ival,
    jit,
    pp_obj,
    pretty_print,
)

T = TypeVar("T")
R = TypeVar("R", covariant=True)
P = ParamSpec("P")


class plot_style(eqx.Module):
    color: tuple[flike, flike, flike] = (1.0, 0.0, 0.0)
    alpha: flike = 1.0

    def clip(self) -> "plot_style":
        clamp_f = lambda x: lax.min(lax.max(x, 0.001), 0.999)
        r, g, b = self.color
        a = self.alpha
        return plot_style((clamp_f(r), clamp_f(g), clamp_f(b)), clamp_f(a))


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
                x=jnp.array(x),
                y=jnp.array(y),
                style=style,
            )
        )

    def clip(self):

        return plot_point(self.x, self.y, self.style.clip())

    def __call__(self, ctx: "plot_ctx"):
        return ctx.point_batched(batched.create(self))


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
            return lax.dynamic_update_slice(x, y, (self.idx,))

        return plot_ctx(
            idx=self.idx + len(p),
            limit=self.limit,
            points=batched_treemap(update_one, self.points, p),
        )

    def __add__(self, f: "plotable") -> "plot_ctx":
        assert isinstance(f, batched)
        f = f
        while isinstance((uf := f.unflatten()), batched):
            f = uf

        f_ = cast_unchecked[batched[Callable[[plot_ctx], plot_ctx]]]()(f)

        if isinstance(uf, plot_point):
            return self.point_batched(cast_unchecked_(f))
        ctx = f_.thread(self)
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
        limit = int(self.limit)

        def to_python(x) -> list[float]:
            ans = np.array(x).tolist()
            assert isinstance(ans, list)
            return ans  # type: ignore

        _points = self.points.unflatten()
        X = to_python(_points.x)
        Y = to_python(_points.y)

        R, G, B = map(to_python, _points.style.color)
        A = to_python(_points.style.alpha)

        assert isinstance(m.points, list)
        assert isinstance(m.colors, list)

        ans: list[Point] = []
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

    @overload
    def __get__(
        self, instance: None, owner: type | None = None, /
    ) -> "plotmethod[T, P]": ...

    @overload
    def __get__(
        self, instance: T, owner: type | None = None, /
    ) -> Callable[P, Callable[[plot_ctx], plot_ctx]]: ...

    def __get__(self, instance: T | None, owner: type | None = None) -> Any:
        if instance is None:
            return self
        else:

            def inner(*args: P.args, **kwargs: P.kwargs):
                return plotmethod_(self.f, instance, args, kwargs)

            return inner
