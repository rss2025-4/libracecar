import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax import lax, random
from jaxtyping import Array, Float
from numpyro.distributions import constraints

from libracecar.plot import plot_ctx, plot_point, plot_style, plotable

from .batched import batched
from .utils import (
    cast_unchecked_,
    debug_print,
    flike,
    fpair,
    fval,
    jit,
    numpyro_param,
    pformat_dataclass,
    pp_obj,
    pretty_print,
)


class cone_location(eqx.Module):
    loc: fpair

    def __repr__(self):
        return pformat_dataclass(self).format()

    def plot(self, style: plot_style = plot_style()) -> plotable:
        return plot_point.create(self.loc, style)


class cone_dist(eqx.Module):
    mean: Float[Array, "2"]
    scale_tril: Float[Array, "2 2"]

    expected_noise: fval

    def covariance(self):
        return self.scale_tril @ self.scale_tril.T

    @staticmethod
    def from_mean_cov(
        mean: Float[Array, "2"], cov: Float[Array, "2 2"], expected_noise: fval
    ):
        return cone_dist(mean, jnp.linalg.cholesky(cov), expected_noise)

    def sample(self, rng_key: Array | None = None) -> tuple[cone_location, fval]:
        cone_d = dist.MultivariateNormal(
            cast_unchecked_(self.mean), scale_tril=self.scale_tril
        )
        sample = numpyro.sample("cone", cone_d, rng_key=rng_key)
        cl = cone_location(jnp.array(sample))

        noise_d = dist.TruncatedNormal(
            loc=cast_unchecked_(self.expected_noise), scale=0.1, low=0.1
        )
        noise = numpyro.sample("noise", cast_unchecked_(noise_d), rng_key=rng_key)
        return cl, jnp.array(noise)

    def __repr__(self):
        return pformat_dataclass(self).format()

    def plot(
        self, n_points: int, rng_key=random.PRNGKey(0), style: plot_style = plot_style()
    ) -> plotable:
        return jax.vmap(
            lambda key: batched.create(self.sample(rng_key=key)[0].plot(style))
        )(random.split(rng_key, num=n_points))


def normal(a, b) -> dist.Distribution:
    return dist.Normal(a, b)


class _compute_posterior_ret(eqx.Module):
    posterior: cone_dist
    losses: Array

    def __repr__(self):
        return pformat_dataclass(self).format()


@jit
def compute_posterior(
    prior: cone_dist, observed: cone_location, rng=random.PRNGKey(0)
) -> _compute_posterior_ret:

    # debug_print("prior", prior)
    # debug_print("observed", observed)

    def guide():
        d = cone_dist(
            numpyro_param("mean", prior.mean),
            # numpyro_param("mean", jnp.array([0.0, 0.0])),
            # numpyro_param(
            #     "scale_tril",
            #     jnp.linalg.cholesky(jnp.eye(2)),
            #     constraints.lower_cholesky,
            # ),
            numpyro_param("scale_tril", prior.scale_tril, constraints.lower_cholesky),
            numpyro_param(
                "expected_noise",
                prior.expected_noise + 0.01,
                constraints.greater_than_eq(0.1),
            ),
        )
        _ = d.sample()

    def model():
        cone_truth, noise = prior.sample()
        # debug_print("ground_truth.loc", ground_truth.loc)
        obs_dist = dist.MultivariateNormal(
            cast_unchecked_(cone_truth.loc), covariance_matrix=jnp.eye(2) * noise**2
        )
        numpyro.sample("_obs", obs_dist, obs=observed.loc)

    optimizer = optax.adamw(learning_rate=0.1)
    svi = numpyro.infer.SVI(
        model, guide, optimizer, loss=numpyro.infer.Trace_ELBO(num_particles=20)
    )

    svi_result = svi.run(rng, 300, progress_bar=False)
    params = svi_result.params

    return _compute_posterior_ret(
        cone_dist(
            params["mean"],
            params["scale_tril"],
            params["expected_noise"],
        ),
        svi_result.losses,
    )
