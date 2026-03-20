from typing import NamedTuple

import jax
import jax.numpy as jnp

import bitboard
from consts import BITBOARD_DTYPE
from state import State

WHITE_ROAD = 1
BLACK_ROAD = 2
NO_RESERVES = 4
FULL_BOARD = 8
REPETITION = 16


# TODO: Prioritize roads > flat wins
# TODO: Prioritize player who just played the move


def terminal_flags(state: State, hashes: jax.Array) -> jax.Array:
    flags = jnp.uint8()

    # Roads
    white_road = road(state.size, state.road_bb & state.white_bb)
    black_road = road(state.size, state.road_bb & state.black_bb)
    flags |= white_road * WHITE_ROAD
    flags |= black_road * BLACK_ROAD

    # Reserves
    no_reserves = (state.white_caps == 0 & state.white_stones == 0) | (state.black_caps == 0 & state.black_caps == 0)
    flags |= no_reserves * NO_RESERVES

    # Board fill
    any_color = state.white_bb | state.black_bb
    full_board = ~any_color & bitboard.board(state.size) == 0
    flags |= full_board * FULL_BOARD

    # 3-fold Repetition
    repetition = jnp.count_nonzero(hashes == state.zobrist_hash) >= 2  # noqa: PLR2004
    flags |= repetition * REPETITION

    return flags


def road(size: int, road_bb: jax.Array) -> jax.Array:
    bottom = bitboard.row(size)
    top = bottom << (size - 1) * size
    right = bitboard.col(size)
    left = right << size - 1

    class FloodState(NamedTuple):
        prev: jax.Array
        now: jax.Array

    def cond_fn(state: FloodState) -> jax.Array:
        return state.prev != state.now

    def body_fn(state: FloodState) -> FloodState:
        return FloodState(prev=state.now, now=bitboard.flood(size, state.now, road_bb))

    bottom_after_flood = jax.lax.while_loop(cond_fn, body_fn, FloodState(prev=BITBOARD_DTYPE(0), now=bottom)).now
    right_after_flood = jax.lax.while_loop(cond_fn, body_fn, FloodState(prev=BITBOARD_DTYPE(0), now=right)).now

    return (top & bottom_after_flood) | (left & right_after_flood)
