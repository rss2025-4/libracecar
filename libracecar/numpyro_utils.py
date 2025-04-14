from typing import (
    Callable,
    ParamSpec,
    TypeVar,
)

import jax
import numpyro
import numpyro.distributions as dist
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, ArrayLike
from numpyro.distributions import Distribution, constraints

from .batched import batched, batched_vmap
from .utils import cast_unchecked_, flike, jit

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def prng_key_():
    ans = numpyro.prng_key()
    assert ans is not None
    return ans


def numpyro_param(
    name: str,
    init_value: ArrayLike,
    constraint: constraints.Constraint = constraints.real,
) -> Array:
    ans = numpyro.param(name, init_value, constraint=constraint)
    assert ans is not None
    return jnp.array(ans)


def numpyro_scope_fn(prefix: str, f: Callable[[], R]) -> R:
    with numpyro.handlers.scope(prefix=prefix):
        return f()


def vmap_seperate_seed(f: Callable[P, R], axis_size: int) -> Callable[P, R]:

    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        seeds = random.split(prng_key_(), num=axis_size)

        def inner2(key: Array, *args: P.args, **kwargs: P.kwargs) -> R:
            with numpyro.handlers.seed(rng_seed=key):
                return f(*args, **kwargs)

        return jax.vmap(inner2)(seeds, *args, **kwargs)

    return inner


def jit_with_seed(f: Callable[P, R]) -> Callable[P, R]:
    @jit
    def inner(key: Array, *args: P.args, **kwargs: P.kwargs):
        with numpyro.handlers.seed(rng_seed=key):
            return f(*args, **kwargs)

    def inner2(*args: P.args, **kwargs: P.kwargs):
        return inner(prng_key_(), *args, **kwargs)

    return inner2


def batched_vmap_with_rng(f: Callable[[T], R], a1: batched[T], /) -> batched[R]:
    bds = a1.batch_dims()
    keys = batched.create(random.split(prng_key_(), bds), bds)

    def inner(key: Array, a1: T) -> R:
        with numpyro.handlers.seed(rng_seed=key):
            return f(a1)

    return batched_vmap(inner, keys, a1)


class batched_dist(Distribution):

    _wrapped: batched[Distribution]
    pytree_data_fields = ("_wrapped",)

    def __init__(self, wrapped: batched[Distribution]):
        def _check(x: Distribution):
            assert x.batch_shape == ()
            return x.event_shape

        event_shape = wrapped.static_map(_check)
        self._wrapped = wrapped
        super().__init__(
            batch_shape=wrapped.batch_dims(),
            event_shape=event_shape,
        )

    def _apply(self, f: Callable[[Distribution], R]) -> R:
        return self._wrapped.map(f).unflatten()

    def _todo(self, *args, **kwargs):
        assert False

    @property
    def mean(self):
        return self._apply(lambda x: x.mean)

    @property
    def variance(self):
        return self._apply(lambda x: x.variance)

    def cdf(self, value: ArrayLike):
        return batched_vmap(
            lambda me, v: me.cdf(v),
            self._wrapped,
            batched.create(value, self.batch_shape),
        ).unflatten()

    sample_with_intermediates = _todo

    def sample(self, key, sample_shape=()):
        return self._wrapped.map_with_rng(
            key,
            lambda d, k: d.sample(k),
        ).unflatten()

    def log_prob(self, value):
        return batched_vmap(
            lambda me, v: me.log_prob(v),
            self._wrapped,
            batched.create(value, self.batch_shape, broadcast=True),
        ).unflatten()

    @property
    def support(self):
        return self._wrapped.static_map(lambda x: x.support)

    @property
    def is_discrete(self):
        return self._wrapped.static_map(lambda x: x.is_discrete)


def normal_(loc: flike, scale: flike) -> dist.Distribution:
    return dist.Normal(loc=cast_unchecked_(loc), scale=cast_unchecked_(scale))


def trunc_normal_(
    loc: flike, scale: flike, low: flike | None = None, high: flike | None = None
) -> dist.Distribution:
    return dist.TruncatedNormal(
        loc=cast_unchecked_(loc), scale=cast_unchecked_(scale), low=low, high=high
    )


def mixturesamefamily_(
    parts_with_logits: batched[tuple[dist.Distribution, flike]],
) -> dist.Distribution:

    parts, weights = parts_with_logits.reshape(-1).split_tuple()

    mixing_dist = dist.Categorical(logits=weights.uf)
    assert isinstance(mixing_dist, dist.Distribution)

    return dist.MixtureSameFamily(mixing_dist, batched_dist(parts))
