import chex
import jax
import jax.numpy as jnp

from consts import BITBOARD_DTYPE, BITBOARD_IINFO


def bb_to_mask(bb: jax.Array) -> jax.Array:
    chex.assert_shape(bb, ())
    assert bb.dtype == jnp.uint64
    indices = jnp.arange(64, dtype=jnp.uint8)
    return ((bb >> indices) & 1).astype(jnp.bool)


def mask_to_bb(mask: jax.Array) -> jax.Array:
    flattened = mask.flatten().astype(jnp.uint64)
    shifts = jnp.arange(flattened.shape[0], dtype=jnp.uint8)
    return jnp.sum(flattened << shifts)


def row(size: int) -> jax.Array:
    return BITBOARD_DTYPE(BITBOARD_IINFO.max) >> (BITBOARD_IINFO.bits - size)


def col(size: int) -> jax.Array:
    acc = BITBOARD_DTYPE(1)
    # jax.lax.scan ?
    for _ in range(1, size):
        acc |= acc << size
    return acc


def board(size: int) -> jax.Array:
    return BITBOARD_DTYPE(BITBOARD_IINFO.max) >> (BITBOARD_IINFO.bits - size * size)


def flood(size: int, bitboard: jax.Array, allowed: jax.Array) -> jax.Array:
    right = bitboard >> 1
    left = bitboard << 1
    up = bitboard << size
    down = bitboard >> size
    attempt = bitboard | right | left | up | down
    return attempt & allowed
