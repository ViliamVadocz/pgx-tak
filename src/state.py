from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify  # https://docs.jax.dev/en/latest/debugging/checkify_guide.html

from consts import DEFAULT_CAPS, DEFAULT_SIZE, DEFAULT_STONES, MAX_STACK_SIZE, check_size


class State(NamedTuple):
    size: int = DEFAULT_SIZE
    starting_stones: int = DEFAULT_STONES
    starting_caps: int = DEFAULT_CAPS

    white_to_move: jax.Array = jnp.bool(True)
    stacks: jax.Array = jnp.ones(DEFAULT_SIZE * DEFAULT_SIZE, dtype=jnp.uint64)

    # bitboards
    white_bb: jax.Array = jnp.uint64(0)
    black_bb: jax.Array = jnp.uint64(0)
    road_bb: jax.Array = jnp.uint64(0)
    noble_bb: jax.Array = jnp.uint64(0)

    # reserves
    white_stones: jax.Array = jnp.uint8(DEFAULT_STONES)
    black_stones: jax.Array = jnp.uint8(DEFAULT_STONES)
    white_caps: jax.Array = jnp.uint2(DEFAULT_CAPS)
    black_caps: jax.Array = jnp.uint2(DEFAULT_CAPS)
