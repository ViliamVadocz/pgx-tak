import typing

import jax
import jax.numpy as jnp

from bitboard import bb_to_mask
from consts import MAX_SIZE, MAX_STACK_SIZE

if typing.TYPE_CHECKING:
    from state import State

HASH_DTYPE = jnp.int64
HASH_MIN = jax.dtypes.iinfo(HASH_DTYPE).min
HASH_MAX = jax.dtypes.iinfo(HASH_DTYPE).max

key = jax.random.key(4963378918276368795)
(
    to_move_key,
    board_size_key,
    capstone_key,
    wall_key,
    stack_key,
) = jax.random.split(key, 5)

WHITE_TO_MOVE = jax.random.randint(to_move_key, (), HASH_MIN, HASH_MAX, HASH_DTYPE)
BOARD_SIZE = jax.random.randint(board_size_key, (MAX_SIZE + 1,), HASH_MIN, HASH_MAX, HASH_DTYPE)
CAPSTONE = jax.random.randint(capstone_key, (MAX_SIZE * MAX_SIZE,), HASH_MIN, HASH_MAX, HASH_DTYPE)
WALL = jax.random.randint(wall_key, (MAX_SIZE * MAX_SIZE,), HASH_MIN, HASH_MAX, HASH_DTYPE)
STACK_COLOR = jax.random.randint(stack_key, (MAX_SIZE * MAX_SIZE, MAX_STACK_SIZE, 2), HASH_MIN, HASH_MAX, HASH_DTYPE)


def get_hash(state: "State") -> jax.Array:
    x = BOARD_SIZE[state.size]
    x = jax.lax.cond(state.white_to_move, lambda: x ^ WHITE_TO_MOVE, lambda: x)
    caps = bb_to_mask(state.road_bb & state.noble_bb)
    walls = bb_to_mask(~state.road_bb & state.noble_bb)
    x ^= jnp.bitwise_xor.reduce(jnp.where(caps, CAPSTONE, 0))
    x ^= jnp.bitwise_xor.reduce(jnp.where(walls, WALL, 0))

    stack_heights = MAX_STACK_SIZE - jax.lax.clz(state.stacks)
    indices = jnp.arange(MAX_STACK_SIZE, dtype=jnp.uint8)
    white = ((state.stacks[:, None] >> indices) & 1).astype(jnp.bool)
    in_stack = stack_heights[:, None] > indices
    board_size = state.stacks.shape[0]
    mask = jnp.dstack([white & in_stack, ~white & in_stack]).reshape((board_size, MAX_STACK_SIZE, 2))

    x ^= jnp.bitwise_xor.reduce(jnp.where(mask, STACK_COLOR[:board_size], 0), axis=None)
    return x
