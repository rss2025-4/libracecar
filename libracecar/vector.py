import equinox as eqx
import numpy as np
from jax import Array, lax
from jax import numpy as jnp
from jaxtyping import ArrayLike, Complex64, Float

from .plot import plot_point, plot_style, plotable
from .utils import bval, flike, fval, jit, pformat_repr


class vec(eqx.Module):
    _v: Complex64[Array, ""]

    @property
    def x(self):
        return self._v.real

    @property
    def y(self):
        return self._v.imag

    def conj(self):
        return vec(self._v.conj())

    @staticmethod
    def create(x: flike, y: flike):
        x = jnp.array(x, dtype=np.float32)
        y = jnp.array(y, dtype=np.float32)
        assert x.shape == ()
        assert y.shape == ()
        return vec(lax.complex(x, y))

    @staticmethod
    def from_arr(x: Float[ArrayLike, "2"]):
        x = jnp.array(x, dtype=np.float32)
        assert x.shape == (2,)
        return vec(lax.complex(x[0], x[1]))

    def as_arr(self):
        return jnp.array([self.x, self.y])

    def __add__(self, other: "vec"):
        return vec(self._v + other._v)

    def __sub__(self, other: "vec"):
        return vec(self._v - other._v)

    def __mul__(self, other: flike | "vec") -> "vec":
        if isinstance(other, vec):
            return vec(self._v * other._v)
        return vec(self._v * jnp.array(other))

    def __truediv__(self, other: flike | "vec") -> "vec":
        if isinstance(other, vec):
            return vec(self._v / other._v)
        return vec(self._v / jnp.array(other))

    __rmul__ = __mul__

    def __neg__(self):
        return vec(-self._v)

    def __iter__(self):
        return iter((self.x, self.y))

    def which_side(self, v: "vec") -> bval:
        """
        which side is v on wrt the line of self ?

        if self is to the right,
        above -> true, below -> false
        """
        ans = v * self.conj()
        return ans.y > 0

    def plot(self, style: plot_style = plot_style()) -> plotable:
        return plot_point.create((self.x, self.y), style)

    __repr__ = pformat_repr


class unitvec(vec):

    @classmethod
    @property
    def one(cls):
        return unitvec(lax.complex(1.0, 0.0))

    @classmethod
    @property
    def i(cls):
        return unitvec(lax.complex(0.0, 1.0))

    def invert(self):
        return unitvec(self._v.conj())

    @staticmethod
    def from_angle(a: flike):
        assert jnp.shape(a) == ()
        return unitvec(lax.complex(jnp.cos(a), jnp.sin(a)))

    @jit
    def to_angle(self) -> fval:
        return jnp.arctan2(self.y, self.x)

    def project(self, v: vec):
        rot = v * self.invert()
        ans = vec.create(rot.x, 0.0)
        return ans * self

    def mul_unit(self, other: "unitvec"):
        assert isinstance(other, unitvec)
        return unitvec(self._v * other._v)
