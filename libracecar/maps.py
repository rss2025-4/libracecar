import math
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from _liblocalization_cpp import distance_2d as _distance_2d
from jax import lax
from jaxtyping import Array, Bool, Float
from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion

from libracecar.batched import batched, batched_zip
from libracecar.plot import plot_ctx, plot_point, plot_style, plotable, plotmethod
from libracecar.specs import position
from libracecar.utils import (
    bval,
    ensure_not_weak_typed,
    flike,
    fpair,
    fval,
    ival,
    jit,
    pformat_repr,
    round_clip,
    tree_at_,
    tree_select,
)
from libracecar.vector import unitvec, vec


class GridMeta(eqx.Module):
    w: int = eqx.field(static=True)
    h: int = eqx.field(static=True)

    # The map resolution [m/cell]
    res: float = eqx.field(static=True)

    # https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/MapMetaData.html
    # The origin of the map [m, m, rad].  This is the real-world pose of the
    # cell (0,0) in the map.
    origin_to_pixel_zero_meters: position

    @jit
    def to_pixels(self, origin_to_pos_meters: position):
        pixel_zero_to_pos = (
            self.origin_to_pixel_zero_meters.invert_pose() + origin_to_pos_meters
        )
        return position(
            pixel_zero_to_pos.tran / self.res,
            pixel_zero_to_pos.rot,
        )

    @jit
    def from_pixels(self, pixel_zero_to_pos_pixels: position):
        pixel_zero_to_pos = position(
            pixel_zero_to_pos_pixels.tran * self.res,
            pixel_zero_to_pos_pixels.rot,
        )
        origin_to_pos = self.origin_to_pixel_zero_meters + pixel_zero_to_pos
        return origin_to_pos

    @jit
    def plot_from_pixels_vec(self, ctx: plot_ctx) -> plot_ctx:
        def process_point(p: plot_point):
            ans = self.from_pixels(position.create((p.x, p.y), 0.0))
            return tree_at_(lambda p: (p.x, p.y), p, (ans.x, ans.y))

        return tree_at_(lambda ctx: ctx.points, ctx, ctx.points.map(process_point))


class Grid(eqx.Module):
    meta: GridMeta

    is_obstacle: Bool[Array, "w h"]

    @property
    def w(self):
        return self.meta.w

    @property
    def h(self):
        return self.meta.h

    @property
    def res(self):
        return self.meta.res

    @staticmethod
    def create(msg: OccupancyGrid):
        h = msg.info.height
        w = msg.info.width
        res = msg.info.resolution
        data = np.array(msg.data)
        assert data.shape == (h * w,)

        # from TAs
        origin_p = msg.info.origin.position
        origin_o = msg.info.origin.orientation
        origin_o = euler_from_quaternion(
            (origin_o.x, origin_o.y, origin_o.z, origin_o.w)
        )
        origin = (origin_p.x, origin_p.y, origin_o[2])

        meta = GridMeta(
            h=h,
            w=w,
            res=res,
            origin_to_pixel_zero_meters=position.create(origin[:2], origin[2]),
        )
        return Grid(
            meta=meta,
            # is_obstacle=jnp.array(data.reshape(h, w).T > 0),
            is_obstacle=jnp.array(data.reshape(h, w).T != 0),
        )

    __repr__ = pformat_repr

    def plot(self, style: plot_style = plot_style()) -> plotable:
        @plotmethod
        def inner(is_obstacle: bval, ctx: plot_ctx, i: ival, j: ival) -> plot_ctx:
            return tree_select(
                is_obstacle,
                on_true=ctx + plot_point.create((i, j), style),
                on_false=ctx,
            )

        return batched.create(self.is_obstacle, (self.w, self.h)).enumerate(inner)


class precomputed_point(eqx.Module):
    dist: fval
    angle: unitvec

    __repr__ = pformat_repr


class precomputed_map(eqx.Module):
    grid: Grid

    points: batched[precomputed_point]

    __repr__ = pformat_repr

    @property
    def res(self):
        return self.grid.res


def _distance_2d_cb(w: int, h: int, data: Float[Array, "hw"]):
    print("_distance_2d_cb: callback", w, h)
    data_np = np.array(data, np.float64)
    return _distance_2d(data_np, w, h, 1.0, 0.0)


def distance_2d(grid: Grid) -> Float[Array, "w h"]:
    data_flat = grid.is_obstacle.T.flatten()
    data_flat = jax.vmap(
        lambda x: lax.select(
            x,
            # occupied
            on_true=0.0,
            # free
            on_false=99999.0,
        )
    )(data_flat)

    ans: Array = jax.pure_callback(
        partial(_distance_2d_cb, w=grid.w, h=grid.h),
        result_shape_dtypes=jax.ShapeDtypeStruct(data_flat.shape, jnp.float32),
        data=data_flat,
    )
    return ans.reshape(grid.h, grid.w).T


def precompute_point(grid: Grid, i: ival, j: ival):

    patch_len = 9

    assert patch_len % 2 == 1

    pad_len = patch_len // 2

    is_obstacle = jnp.pad(
        grid.is_obstacle,
        ((pad_len, pad_len), (pad_len, pad_len)),
        mode="constant",
        constant_values=True,
    )

    patch = lax.dynamic_slice(
        is_obstacle, start_indices=(i, j), slice_sizes=(patch_len, patch_len)
    )

    def each_angle(a: unitvec):

        def each_patch_point(p_i: fval, p_j: fval, p: bval):
            return a.which_side(vec.create(p_i, p_j)) & p

        idxs = batched.arange(-pad_len, pad_len + 1)

        pairs = idxs.map(lambda x: idxs.map(lambda y: (x, y))).unflatten()

        patch_points = batched_zip(pairs, batched.create(patch, patch.shape)).tuple_map(
            lambda coord, item: each_patch_point(*coord, item)
        )
        return jnp.sum(patch_points.unflatten())

    angles_ = jnp.linspace(0, 2 * math.pi, 33)[:-1]
    angles = batched.create(angles_, angles_.shape).map(unitvec.from_angle)
    angle_scores = angles.map(each_angle)

    ans_angle = angles[jnp.argmax(angle_scores.unflatten())].unwrap()

    return precomputed_point(
        angle=ans_angle,
        dist=jnp.array(0.0),  # placeholder
    )


@jit
def precompute(grid: Grid) -> precomputed_map:

    ans_ = batched.arange(grid.w).map(
        lambda i: batched.arange(grid.h).map(
            #
            lambda j: precompute_point(grid, i, j)
        )
    )
    ans = ans_.unflatten()

    distance_2d_res = distance_2d(grid)
    ans = tree_at_(lambda x: x.unflatten().dist, ans, distance_2d_res)

    return ensure_not_weak_typed(
        precomputed_map(
            grid=grid,
            points=ans,
        )
    )


class _trace_ray_res(eqx.Module):
    coord: vec
    angle: unitvec

    dist: fval
    distance_to_nearest: fval
    hit_pos: vec
    data: precomputed_point

    @plotmethod
    def plot(self, ctx: plot_ctx, truth: fval | None = None) -> plot_ctx:

        if truth is not None:
            ctx += (self.coord + self.angle * truth).plot(
                plot_style(color=(1.0, 1.0, 0.0))
            )
            return ctx

        normal = self.data.angle * unitvec.from_angle(math.pi / 2)

        ctx1 = ctx
        ctx1 += (self.hit_pos + normal).plot(plot_style(color=(1.0, 0.0, 0.0)))
        ctx1 += (self.hit_pos - normal).plot(plot_style(color=(0.0, 1.0, 0.0)))

        ctx2 = ctx
        ctx2 += self.hit_pos.plot(plot_style(color=(0.0, 0.0, 1.0)))

        return tree_select(self.distance_to_nearest > 2.0, on_false=ctx1, on_true=ctx2)


class _loop_state(eqx.Module):
    coord: vec
    data: precomputed_point
    distance_to_nearest: fval


def trace_ray(
    map: precomputed_map, coord: vec, angle: unitvec, eps: flike = 1.0
) -> _trace_ray_res:

    # print("trace_ray", map)
    # print("angles", map.points[:, :10])

    coord_s = lax.stop_gradient(coord)
    angle_s = lax.stop_gradient(angle)

    def _create_state(coord: vec):
        i, j = coord
        r_i = round_clip(i, 0, map.grid.w)
        r_j = round_clip(j, 0, map.grid.h)
        r_err = jnp.linalg.norm(jnp.array([i - r_i, j - r_j]))

        data = map.points[r_i, r_j].unwrap()
        return _loop_state(
            coord=coord,
            data=data,
            distance_to_nearest=jnp.maximum(data.dist - r_err - eps, 0),
        )

    def step(s: _loop_state):
        return _create_state(s.coord + angle_s * s.distance_to_nearest)

    # loop_res, _ = lax.scan(
    #     lambda c, _: (step(c), None),
    #     init=_create_state(coord_s),
    #     xs=None,
    #     length=16,
    # )

    # unrolled is faster (2025-4, Alan)
    loop_res = _create_state(coord_s)
    for _ in range(16):
        loop_res = step(loop_res)

    ans_s = loop_res.coord

    # make this differentiable

    line_ang_s = loop_res.data.angle

    line_ang_s = tree_select(
        jnp.abs((line_ang_s * angle_s.invert()).x) < 0.99,
        on_true=line_ang_s,
        on_false=line_ang_s.mul_unit(unitvec(lax.complex(0.0, 1.0))),
    )

    coord_relative_to_line = (coord - ans_s) * line_ang_s.invert()
    transformed_ang = angle * line_ang_s.invert()

    ans_dist = -coord_relative_to_line.y / transformed_ang.y

    ans = coord + angle * ans_dist

    # # debug_print("ans:", ans_s, ans)
    # # debug_print("line_ang_s", line_ang_s, coord)
    # check(
    #     jnp.allclose(ans_s._v, ans._v),
    #     "liblocalization.map, {} {} {}",
    #     ans_s,
    #     ans,
    #     transformed_ang,
    # )

    ans = ans_s + (ans - lax.stop_gradient(ans))
    ans_dist = jnp.abs((ans_s - coord)._v) + (ans_dist - lax.stop_gradient(ans_dist))

    return _trace_ray_res(
        coord=coord,
        angle=angle,
        dist=ans_dist,
        distance_to_nearest=loop_res.distance_to_nearest,
        hit_pos=ans,
        data=loop_res.data,
    )
