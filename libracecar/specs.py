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
from jaxtyping import ArrayLike, Float, jaxtyped

from .batched import batched, batched_vmap, batched_zip
from .plot import plot_ctx, plot_point, plot_style, plotable, plotmethod
from .utils import (
    flike,
    fpair,
    fval,
    io_callback_,
    ival,
    pformat_dataclass,
    pformat_repr,
    pp_obj,
    pretty_print,
    safe_select,
)

turn_angle_limit = math.pi / 8
min_turn_radius = 1.0


class position(eqx.Module):
    coord: fpair
    heading: fval

    @staticmethod
    def zero():
        return position(jnp.array([0.0, 0.0]), jnp.array(0.0))

    @staticmethod
    def translation(coord: Float[ArrayLike, "2"]):
        return position(jnp.array(coord), jnp.array(0.0))

    def __add__(self, p: "position"):
        rotated_p = lax.complex(p.coord[0], p.coord[1]) * lax.complex(
            jnp.cos(self.heading), jnp.sin(self.heading)
        )
        return position(
            coord=self.coord + jnp.array([rotated_p.real, rotated_p.imag]),
            heading=self.heading + p.heading,
        )

    def invert_pose(self):
        rotated_coord = lax.complex(-self.coord[0], -self.coord[1]) * lax.complex(
            jnp.cos(self.heading), jnp.sin(self.heading)
        )
        return position(
            coord=jnp.array([rotated_coord.real, rotated_coord.imag]),
            heading=-self.heading,
        )

    def plot(self, style: plot_style = plot_style()) -> plotable:
        return plot_point.create(self.coord, style)

    def _pretty_print(self):
        return pp_obj("position", pretty_print(self.coord), pretty_print(self.heading))

    def __repr__(self):
        return self._pretty_print().format()


@jaxtyped(typechecker=typechecker)
class path_segment(eqx.Module):
    angle: fval
    length: fval

    def __init__(self, angle: flike, length: flike):
        self.angle = jnp.array(angle)
        self.length = jnp.array(length)

    def clip(self) -> "path_segment":
        return path_segment(
            angle=jnp.clip(self.angle, -turn_angle_limit, turn_angle_limit),
            length=self.length,
        )

    @jaxtyped(typechecker=typechecker)
    def move(self, old: position) -> position:
        # ang=pi/8 ==> circle of radius 1
        # ang=pi/8 , dist=pi/2 ==> turn=pi/2

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

    __repr__ = pformat_repr

    @plotmethod
    def plot(self, ctx: plot_ctx, start: position, style: plot_style = plot_style()):
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


T = TypeVar("T")


class path(eqx.Module):
    parts: batched[path_segment]

    @jaxtyped(typechecker=typechecker)
    def move(self, p: position | None = None) -> tuple[position, batched[position]]:
        if p is None:
            p = position.zero()

        final_pos, pos_intermediats = self.parts.scan(
            lambda p, s: (s.move(p), p), init=p
        )
        return final_pos, pos_intermediats

    def clip(self) -> "path":
        return path(self.parts.map(lambda x: x.clip()))

    def __getitem__(self, idx: int) -> path_segment:
        return self.parts[idx].unwrap()

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @staticmethod
    def from_parts(*segs: path_segment) -> "path":
        return path(batched.stack([batched.create(s) for s in segs]))

    __repr__ = pformat_repr

    def plot(
        self, start: position | None = None, style: plot_style = plot_style()
    ) -> plotable:
        _, seg_starts = self.move(start)
        return batched_vmap(lambda p, s: p.plot(s, style), self.parts, seg_starts)
