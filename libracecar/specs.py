"""
properties of the car, and functions to compute car states after actions
"""

from __future__ import annotations

import math
from typing import TypeVar

import equinox as eqx
import jax
import numpy as np
import tf_transformations
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseArray,
    Quaternion,
    TransformStamped,
    Twist,
    Vector3,
)
from jax import Array, lax
from jax import numpy as jnp
from jaxtyping import ArrayLike, Float
from tf_transformations import euler_from_quaternion

from .batched import batched, batched_vmap, vector
from .jax_utils import divide_x_at_zero
from .plot import plot_ctx, plot_point, plot_style, plotable, plotmethod
from .utils import (
    cast_unchecked_,
    flike,
    fval,
    jit,
    lazy,
    pformat_repr,
    pp_obj,
    pretty_print,
    safe_select,
)
from .vector import unitvec, vec

turn_angle_limit = math.pi / 8
min_turn_radius = 1.0


class position(eqx.Module):
    """a 2d pose"""

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
    def _process_trans(trans: Vector3 | Point) -> tuple[float, float]:
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

    @staticmethod
    def from_ros_pose(pose: Pose) -> lazy["position"]:
        return lazy(
            position.create,
            position._process_trans(pose.position),
            position._process_quat(pose.orientation),
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
    def _from_ros_pose_arr(xs, ys, thetas, n) -> vector[position]:
        args = vector(n, batched.create((xs, ys, thetas), (len(xs),)))
        return args.map(lambda args: position.create((args[0], args[1]), args[2]))

    @staticmethod
    def from_ros_pose_arr(poses: PoseArray, limit: int = 100) -> lazy[vector[position]]:
        n = len(poses.poses)
        assert n <= limit
        fill = limit - n

        fill_float = [0.0 for _ in range(fill)]

        xs = []
        ys = []
        thetas = []

        for p in poses.poses:
            assert isinstance(p, Pose)
            x, y = position._process_trans(p.position)
            xs.append(x)
            ys.append(y)
            thetas.append(position._process_quat(p.orientation))

        return lazy(
            position._from_ros_pose_arr,
            np.array(xs + fill_float),
            np.array(ys + fill_float),
            np.array(thetas + fill_float),
            np.array(n, dtype=jnp.int32),
        )

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


class twist_t(eqx.Module):
    # pixel / s
    linear: vec
    # rad / s
    angular: fval
    # s
    time: fval

    __repr__ = pformat_repr

    @staticmethod
    def _create(linear_x, linear_y, angular, time):
        return twist_t(vec.create(linear_x, linear_y), angular, time)

    @staticmethod
    def zero():
        return twist_t(vec.create(0, 0), jnp.array(0.0), jnp.array(0.0))

    @staticmethod
    def from_ros(msg: Twist, time: float, res: float) -> lazy["twist_t"]:
        assert msg.linear.z == 0.0

        assert msg.angular.x == 0.0
        assert msg.angular.y == 0.0

        return lazy(
            twist_t._create,
            linear_x=float(msg.linear.x) / res,
            linear_y=float(msg.linear.y) / res,
            angular=float(msg.angular.z),
            time=time,
        )

    def to_position(self) -> position:
        # ang = exp( (angular * t) * i) * linear
        # ang_integal = exp( (angular * t) * i) / (angular * i) * linear
        # ang_integal(0) = 1 / (angular * i) * linear
        # ang_integal(T) = rot / (angular * i) linear

        def n(angular: fval):
            rot = unitvec.from_angle(angular * self.time)
            return rot - unitvec.one

        def d(angular: fval):
            return angular * unitvec.i

        def nonzero_case():
            return n(self.angular) / d(self.angular)

        def zero_case():
            return divide_x_at_zero(n)(self.angular) / divide_x_at_zero(d)(self.angular)

        ans = safe_select(
            jnp.abs(self.angular) <= 1e-5,
            on_false=nonzero_case,
            on_true=zero_case,
        )

        return position(ans * self.linear, unitvec.from_angle(self.angular * self.time))

    def transform(self, transform: position) -> twist_t:

        def inner(t: fval):
            transform_self = twist_t(self.linear, self.angular, t).to_position()
            transform_other = transform.invert_pose() + transform_self + transform
            return transform_other.tran

        _primals_out, tangents_out = jax.jvp(inner, (0.0,), (1.0,))
        assert isinstance(tangents_out, vec)

        return twist_t(tangents_out, self.angular, self.time)


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

    def move(self, old: position) -> position:
        # ang=pi/8 ==> circle of radius 1
        # ang=pi/8 , dist=pi/2 ==> turn=pi/2

        # suppose: move at 1m/s with self.length seconds

        # ang=pi/8, v=1m/s ==> ang_v = 1 rad/s

        twist = twist_t(
            linear=vec.create(1, 0),
            angular=(self.angle / (math.pi / 8)),
            time=self.length,
        )

        return old + twist.to_position()

    __repr__ = pformat_repr

    @plotmethod
    def plot(self, ctx: plot_ctx, start: position, style: plot_style = plot_style()):
        def plot_one(d: fval, ctx: plot_ctx):
            mid = path_segment(
                self.angle,
                lax.select(self.length > 0, d, -d),
            ).move(start)
            ctx += mid.plot_as_point(style)
            return d + 0.1, ctx

        _, ctx = lax.while_loop(
            init_val=(jnp.array(0.0), ctx),
            cond_fun=lambda p: p[0] < jnp.abs(self.length),
            body_fun=lambda p: plot_one(p[0], p[1]),
        )
        _, ctx = plot_one(jnp.abs(self.length), ctx)
        return ctx


class path(eqx.Module):
    parts: batched[path_segment]

    @jit
    def move(self, p: position | None = None) -> tuple[position, batched[position]]:
        if p is None:
            p = position.zero()

        final_pos, pos_intermediats = self.parts.scan(
            lambda p, s: (s.move(p), p), init=p
        )
        return final_pos, pos_intermediats

    def clip(self) -> path:
        return path(self.parts.map(lambda x: x.clip()))

    def __getitem__(self, idx: int) -> path_segment:
        return self.parts[idx].unwrap()

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @staticmethod
    def from_parts(*segs: path_segment) -> path:
        return path(batched.create_stack(segs))

    __repr__ = pformat_repr

    def plot(
        self, start: position | None = None, style: plot_style = plot_style()
    ) -> plotable:
        _, seg_starts = self.move(start)
        return batched_vmap(lambda p, s: p.plot(s, style), self.parts, seg_starts)
