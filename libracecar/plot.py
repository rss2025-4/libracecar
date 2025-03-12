import math
from typing import Callable, Protocol, TypeVar

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

from .utils import (
    batched,
    debug_callback,
    debug_print,
    flatten_n_tree,
    flike,
    fpair,
    fval,
    ival,
    jit,
    pp_obj,
    pretty_print,
)


class plot_style(eqx.Module):
    color: tuple[flike, flike, flike] = (1.0, 0.0, 0.0)
    alpha: flike = 1.0

    def clip(self) -> "plot_style":
        clamp_f = lambda x: lax.min(lax.max(x, 0.001), 0.999)
        r, g, b = self.color
        a = self.alpha
        return plot_style((clamp_f(r), clamp_f(g), clamp_f(b)), clamp_f(a))


class _plot_point(eqx.Module):
    x: fval
    y: fval

    style: plot_style

    def clip(self):
        return _plot_point(self.x, self.y, self.style.clip())


class plot_ctx(eqx.Module):
    idx: ival
    limit: int = eqx.field(static=True)
    points: batched[_plot_point, 0]

    @staticmethod
    def create(limit: int) -> "plot_ctx":
        points_buf = jax.vmap(
            lambda: _plot_point(jnp.array(0.0), jnp.array(0.0), plot_style()),
            axis_size=limit,
        )()
        return plot_ctx(
            idx=jnp.array(0),
            limit=limit,
            points=points_buf,
        )

    @jit
    def point(
        self, loc: tuple[flike, flike] | fpair, style: plot_style = plot_style()
    ) -> "plot_ctx":
        x, y = loc
        ans = _plot_point(
            x=jnp.array(x),
            y=jnp.array(y),
            style=style,
        )
        return self.push_batched(ans)

    @jit
    def push_batched(self, p: batched[_plot_point, ...]) -> "plot_ctx":
        p = flatten_n_tree(p, len(p.x.shape))
        p = jax.vmap(_plot_point.clip)(p)
        lp = len(p.x)

        def update_one(x: Array, y: Array):
            assert len(x) == self.limit
            assert len(y) == lp
            assert x.shape[1:] == y.shape[1:]
            return lax.dynamic_update_slice(
                x, y, (self.idx, *(0 for _ in (x.shape[1:])))
            )

        return plot_ctx(
            idx=self.idx + lp,
            limit=self.limit,
            points=jtu.tree_map(update_one, self.points, p),
        )

    def __add__(self, f: Callable[["plot_ctx"], "plot_ctx"]) -> "plot_ctx":
        return f(self)

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

        X = to_python(self.points.x)
        Y = to_python(self.points.y)

        R, G, B = map(to_python, self.points.style.color)
        A = to_python(self.points.style.alpha)

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
