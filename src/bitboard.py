import chex
import jax
import jax.numpy as jnp


def bb_to_mask(bb: jax.Array) -> jax.Array:
    chex.assert_shape(bb, ())
    assert bb.dtype == jnp.uint64
    indices = jnp.arange(64, dtype=jnp.uint8)
    return ((bb >> indices) & 1).astype(jnp.bool)


def mask_to_bb(mask: jax.Array) -> jax.Array:
    flattened = mask.flatten().astype(jnp.uint64)
    shifts = jnp.arange(flattened.shape[0], dtype=jnp.uint8)
    return jnp.sum(flattened << shifts)
