"""
properties of the car, and functions to compute car states after actions
"""

import math
from typing import TypeVar

import equinox as eqx
import numpy as np
import tf_transformations
from geometry_msgs.msg import Pose, PoseArray, Quaternion, TransformStamped, Vector3
from jax import Array
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float
from tf_transformations import euler_from_quaternion

from libracecar.vector import unitvec, vec

from .batched import batched
from .plot import plot_point, plot_style, plotable
from .utils import (
    cast_unchecked_,
    flike,
    lazy,
    pp_obj,
    pretty_print,
)

turn_angle_limit = math.pi / 8
min_turn_radius = 1.0


class position(eqx.Module):
    tran: vec
    rot: unitvec

    @property
    def x(self):
        return self.tran.x

    @property
    def y(self):
        return self.tran.y

    def as_arr(self) -> Float[Array, "3"]:
        return jnp.array([self.x, self.y, self.rot.to_angle()])

    @staticmethod
    def from_arr(arr: Float[ArrayLike, "3"]) -> "position":
        arr = jnp.array(arr)
        assert arr.shape == (3,)
        return position.create((arr[0], arr[1]), arr[2])

    @staticmethod
    def zero():
        return position(vec.create(0.0, 0.0), unitvec.one)

    @staticmethod
    def translation(coord: Float[ArrayLike, "2"]):
        return position(vec.from_arr(coord), unitvec.one)

    @staticmethod
    def create(
        coord: tuple[flike, flike] | Float[ArrayLike, "2"], heading: flike | None = None
    ):
        coord = jnp.array(coord)
        if heading is None:
            rot = unitvec.one
        else:
            rot = unitvec.from_angle(heading)
        assert coord.shape == (2,)
        return position(vec.from_arr(coord), rot)

    @staticmethod
    def _process_trans(trans: Vector3) -> tuple[float, float]:
        assert trans.z == 0.0
        return (trans.x, trans.y)

    @staticmethod
    def _process_quat(quat: Quaternion) -> float:
        quat_ = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
        assert quat_[0] == 0.0
        assert quat_[1] == 0.0
        return quat_[2]

    @staticmethod
    def from_ros(pose: TransformStamped) -> lazy["position"]:
        return lazy(
            position.create,
            position._process_trans(pose.transform.translation),
            position._process_quat(pose.transform.rotation),
        )

    def to_ros(self) -> Pose:
        x, y, z, w = tf_transformations.quaternion_from_euler(
            0.0, 0.0, float(self.rot.to_angle())
        )
        quat = Quaternion()
        quat.x = float(x)
        quat.y = float(y)
        quat.z = float(z)
        quat.w = float(w)

        ans = Pose()
        ans.orientation = quat
        ans.position.x = float(self.tran.x)
        ans.position.y = float(self.tran.y)

        return ans

    @staticmethod
    def _from_ros_pose_arr(xs, ys, thetas):
        return batched.create((xs, ys, thetas), (len(xs),)).tuple_map(
            lambda x, y, theta: position.create((x, y), theta)
        )

    @staticmethod
    def from_ros_pose_arr(
        poses: PoseArray, limit: int = 100
    ) -> tuple[lazy[batched["position"]], int]:
        n = len(poses.poses)
        assert n <= limit
        fill = limit - n

        fill_float = [0.0 for _ in range(fill)]

        xs = []
        ys = []
        thetas = []

        for p in poses.poses:
            assert isinstance(p, Pose)
            x, y = position._process_trans(cast_unchecked_(p.position))
            xs.append(x)
            ys.append(y)
            thetas.append(position._process_quat(p.orientation))

        ans = lazy(
            position._from_ros_pose_arr, np.array(xs), np.array(ys), np.array(thetas)
        )
        return ans, n

    @staticmethod
    def lazy_zero() -> lazy["position"]:
        return lazy(position.create, (0.0, 0.0), 0.0)

    def __add__(self, p: "position"):
        return position(
            tran=self.tran + p.tran * self.rot,
            rot=self.rot.mul_unit(p.rot),
        )

    def invert_pose(self):
        inv_r = self.rot.invert()
        return position(
            tran=-self.tran * inv_r,
            rot=inv_r,
        )

    def plot_as_point(self, style: plot_style = plot_style()) -> plotable:
        return plot_point.create(self.tran.as_arr(), style)

    def plot_as_seg(self, style: plot_style = plot_style()) -> plotable:
        p1 = plot_point.create(
            self.tran.as_arr(), plot_style(color=style.color, alpha=style.alpha / 5)
        )
        p2 = plot_point.create((self.tran + self.rot).as_arr(), style)
        return batched.stack([p1, p2])

    def _pretty_print(self):
        return pp_obj("position", pretty_print(self.tran), pretty_print(self.rot))

    def __repr__(self):
        return self._pretty_print().format()


# @jaxtyped(typechecker=typechecker)
# class path_segment(eqx.Module):
#     angle: fval
#     length: fval

#     def __init__(self, angle: flike, length: flike):
#         self.angle = jnp.array(angle)
#         self.length = jnp.array(length)

#     def clip(self) -> "path_segment":
#         return path_segment(
#             angle=jnp.clip(self.angle, -turn_angle_limit, turn_angle_limit),
#             length=self.length,
#         )

#     @jaxtyped(typechecker=typechecker)
#     def move(self, old: position) -> position:
#         # ang=pi/8 ==> circle of radius 1
#         # ang=pi/8 , dist=pi/2 ==> turn=pi/2


#         turn = (self.angle / turn_angle_limit) * self.length / min_turn_radius

#         h1 = old.heading
#         h2 = old.heading + turn

#         zero_turn = jnp.abs(turn) <= 1e-8

#         offsetx = safe_select(
#             zero_turn,
#             on_false=lambda: ((jnp.sin(h2) - jnp.sin(h1)) / turn * self.length),
#             on_true=lambda: self.length,
#         )

#         offsety = safe_select(
#             zero_turn,
#             on_false=lambda: ((-jnp.cos(h2) + jnp.cos(h1)) / turn * self.length),
#             on_true=lambda: 0.0,
#         )

#         new_pos = old.coord + jnp.array([offsetx, offsety])

#         return position(
#             coord=new_pos,
#             heading=h2,
#         )

#     __repr__ = pformat_repr

#     @plotmethod
#     def plot(self, ctx: plot_ctx, start: position, style: plot_style = plot_style()):
#         def plot_one(d: fval, ctx: plot_ctx):
#             mid = path_segment(
#                 self.angle,
#                 lax.select(self.length > 0, d, -d),
#             ).move(start)
#             ctx += mid.plot(style)
#             return d + 0.1, ctx

#         _, ctx = lax.while_loop(
#             init_val=(jnp.array(0.0), ctx),
#             cond_fun=lambda p: p[0] < jnp.abs(self.length),
#             body_fun=lambda p: plot_one(p[0], p[1]),
#         )
#         _, ctx = plot_one(jnp.abs(self.length), ctx)
#         return ctx


T = TypeVar("T")


# class path(eqx.Module):
#     parts: batched[path_segment]

#     @jaxtyped(typechecker=typechecker)
#     def move(self, p: position | None = None) -> tuple[position, batched[position]]:
#         if p is None:
#             p = position.zero()

#         final_pos, pos_intermediats = self.parts.scan(
#             lambda p, s: (s.move(p), p), init=p
#         )
#         return final_pos, pos_intermediats

#     def clip(self) -> "path":
#         return path(self.parts.map(lambda x: x.clip()))

#     def __getitem__(self, idx: int) -> path_segment:
#         return self.parts[idx].unwrap()

#     def __len__(self):
#         return len(self.parts)

#     def __iter__(self):
#         return (self[i] for i in range(len(self)))

#     @staticmethod
#     def from_parts(*segs: path_segment) -> "path":
#         return path(batched.stack([batched.create(s) for s in segs]))

#     __repr__ = pformat_repr

#     def plot(
#         self, start: position | None = None, style: plot_style = plot_style()
#     ) -> plotable:
#         _, seg_starts = self.move(start)
#         return batched_vmap(lambda p, s: p.plot(s, style), self.parts, seg_starts)
