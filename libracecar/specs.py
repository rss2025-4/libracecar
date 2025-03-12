"""
properties of the car, and functions to compute car states after actions
"""

import math
from typing import Callable, TypeVar

import equinox as eqx
import jax
import numpy as np
from beartype import beartype as typechecker
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jaxtyping import jaxtyped

from .plot import _plot_point, plot_ctx, plot_style
from .utils import (
    batched,
    check_shape,
    flatten_n_tree,
    flike,
    fpair,
    fval,
    io_callback_,
    ival,
    pp_obj,
    pretty_print,
    safe_select,
    tree_stack,
)

turn_angle_limit = math.pi / 8
min_turn_radius = 1.0


class position(eqx.Module):
    coord: fpair
    heading: fval

    def plot(self, style: plot_style = plot_style()):
        def inner(ctx: plot_ctx) -> plot_ctx:
            return ctx.point(self.coord, style)

        return inner

    def _pretty_print(self):
        return pp_obj("position", pretty_print(self.coord), pretty_print(self.heading))

    def __repr__(self):
        return self._pretty_print().format()


class path_segment(eqx.Module):
    angle: fval
    length: fval

    def __init__(self, angle: flike, length: flike):
        self.angle = jnp.array(angle)
        self.length = jnp.array(length)

    def check_shape(self, *batch_dims: int):
        check_shape(self.angle, *batch_dims)
        check_shape(self.length, *batch_dims)

    def clip(self) -> "path_segment":
        return path_segment(
            angle=jnp.clip(self.angle, -turn_angle_limit, turn_angle_limit),
            length=self.length,
        )

    @jaxtyped(typechecker=typechecker)
    def move(self, old: position) -> position:
        # ang=pi/8 ==> circle of radius 1
        # ang=pi/8 , dist=pi/2 ==> turn=pi/2

        self.check_shape()
        turn = (self.angle / turn_angle_limit) * self.length / min_turn_radius

        h1 = old.heading
        h2 = old.heading + turn

        zero_turn = jnp.abs(turn) <= 1e-8

        offsetx = safe_select(
            zero_turn,
            on_false=lambda: ((jnp.sin(h2) - jnp.sin(h1)) / turn * self.length),
            on_true=lambda: self.length,
        )

        offsety = safe_select(
            zero_turn,
            on_false=lambda: ((-jnp.cos(h2) + jnp.cos(h1)) / turn * self.length),
            on_true=lambda: 0.0,
        )

        new_pos = old.coord + jnp.array([offsetx, offsety])

        return position(
            coord=new_pos,
            heading=h2,
        )

    def _pretty_print(self):
        return pp_obj(
            "path_segment", pretty_print(self.angle), pretty_print(self.length)
        )

    def __repr__(self):
        return self._pretty_print().format()

    def plot_(self, start: position, style: plot_style = plot_style()):
        self.check_shape()

        def inner(ctx: plot_ctx) -> plot_ctx:
            def plot_one(d: fval, ctx: plot_ctx):
                mid = path_segment(
                    self.angle,
                    lax.select(self.length > 0, d, -d),
                ).move(start)
                ctx += mid.plot(style)
                return d + 0.1, ctx

            _, ctx = lax.while_loop(
                init_val=(jnp.array(0.0), ctx),
                cond_fun=lambda p: p[0] < jnp.abs(self.length),
                body_fun=lambda p: plot_one(p[0], p[1]),
            )
            _, ctx = plot_one(jnp.abs(self.length), ctx)
            return ctx

        return inner

    def plot(
        self: batched["path_segment", ...],
        start: batched[position, ...],
        style: plot_style = plot_style(),
    ):
        self, start = flatten_n_tree((self, start), len(start.heading.shape))

        def inner(ctx: plot_ctx) -> plot_ctx:
            ctx, _ = lax.scan(
                lambda ctx, x: (ctx + x[0].plot_(x[1], style), None),
                xs=(self, start),
                init=ctx,
            )
            return ctx

        return inner


T = TypeVar("T")


class path(eqx.Module):
    parts: batched[path_segment, 0]

    def check_shape(self, *batch_dims: int):
        self.parts.check_shape(*batch_dims, -1)

    @jaxtyped(typechecker=typechecker)
    def move(self, p: position | None = None) -> tuple[position, batched[position, 0]]:
        if p is None:
            p = position(coord=jnp.array([0.0, 0.0]), heading=jnp.array(0.0))
        final_pos, pos_intermediats = lax.scan(
            lambda p, s: (s.move(p), p), init=p, xs=self.parts
        )
        return final_pos, pos_intermediats

    def clip(self) -> "path":
        return path(jax.vmap(path_segment.clip)(self.parts))

    def __getitem__(self, idx: int) -> path_segment:
        return jtu.tree_map(lambda x: x[idx], self.parts)

    def __len__(self):
        return len(self.parts.angle)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def map_parts(self, m: Callable[[path_segment], T]) -> batched[T, 0]:
        return jax.vmap(m)(self.parts)

    @staticmethod
    def from_parts(*segs: path_segment) -> "path":
        return path(tree_stack(*segs))

    def _pretty_print(self):
        return pp_obj(
            "path", pretty_print(self.parts.angle), pretty_print(self.parts.length)
        )

    def __repr__(self):
        return self._pretty_print().format()

    def plot_(self, start: position | None = None, style: plot_style = plot_style()):
        _, seg_starts = self.move(start)
        return self.parts.plot(seg_starts, style)

    def plot(
        self: batched["path", ...],
        start: batched[position, ...] | None = None,
        style: plot_style = plot_style(),
    ):
        self, start = flatten_n_tree((self, start), len(self.parts.angle.shape))
        return self.plot_(start, style)
