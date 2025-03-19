import equinox as eqx
from jax import lax
from jax import numpy as jnp

from .utils import flike, fval, jit

"""
models the probability of measuring z as range as a Gaussian
Args:
    z: measured range.
    d: The ground truth distance.
Returns:
    Probability density
"""


class gausian(eqx.Module):
    # Standard deviation
    sigma: flike = 1.0
    # normalization constant such that the Gaussian integrates to 1
    eta: flike = 1.0

    z_max: flike | None = None

    def density(self, z: flike, d: flike) -> fval:
        normalization = 1.0 / jnp.sqrt(2.0 * jnp.pi * (self.sigma**2))
        exponent = jnp.exp(-((z - d) ** 2 / (2.0 * self.sigma**2)))
        ans = self.eta * normalization * exponent
        ans = lax.select(ans > 0, on_true=ans, on_false=0.0)
        if self.z_max is None:
            return ans
        else:
            return lax.select(z < self.z_max, on_true=ans, on_false=0.0)
