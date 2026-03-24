import jax
import jax.numpy as jnp

import bitboard
from consts import BITBOARD_IINFO
from state import State


def num_spread_patterns(size: int) -> int:
    return (1 << size) - 2


def action_mask_channels(size: int) -> int:
    piece_types = 3
    spreads = 4 * num_spread_patterns(size)
    return piece_types + spreads


def legal_action_mask(state: State) -> jax.Array:
    # The action tensor
    #
    # Let N = size of the board, i.e. N = 6 for 6x6 board
    # The action tensor has size N x N x (3 + 4 * (2^N - 2))
    # The first three channels correspond to placements.
    # In order: flats, walls, capstones.
    # The rest of the placements correspond to spreads:
    # Each channel corresponds to a particular pattern,
    # and we repeat the whole pattern sequence for each direction,
    # a total of four times.

    n = state.size
    mask = jnp.zeros((action_mask_channels(n), n, n))

    # Placements
    empty = ~(state.white_bb | state.black_bb)
    empty_mask = bitboard.bb_to_mask(empty)[: n * n].reshape((n, n))
    stones = jax.lax.cond(state.white_to_move, lambda: state.white_stones, lambda: state.black_stones)
    caps = jax.lax.cond(state.white_to_move, lambda: state.white_caps, lambda: state.black_caps)
    mask = jax.lax.cond(stones > 0, lambda: mask.at[0].set(empty_mask).at[1].set(empty_mask), lambda: mask)
    mask = jax.lax.cond(caps > 0, lambda: mask.at[2].set(empty_mask), lambda: mask)

    # Spreads
    my_bb = jax.lax.cond(state.white_to_move, lambda: state.white_bb, lambda: state.black_bb)

    def check_spread(index: jax.Array, pattern: jax.Array, direction: jax.Array) -> jax.Array:
        row = index / n
        col = index % n
        bit = 1 << index
        owned = my_bb & bit > 0
        is_cap = (state.white_caps | state.black_caps) & bit > 0
        left_up_mask = (BITBOARD_IINFO.max >> (index + 1)) << (index + 1)
        right_down_mask = (1 << index) - 1
        row_at_index = bitboard.row(n) << (row * n)
        col_at_index = bitboard.col(n) << col

        ray = jax.lax.switch(
            direction,
            (
                lambda: left_up_mask & row_at_index,  # left
                lambda: left_up_mask & col_at_index,  # up
                lambda: right_down_mask & row_at_index,  # right
                lambda: right_down_mask & col_at_index,  # down
            ),
        )

        hits = ray & state.noble_bb

        distance = jax.lax.switch(direction, (lambda: ...,))  # TODO

        return owned & ()

    # vmap over squares, patterns, and directions
    spreads = jax.vmap(
        jax.vmap(
            jax.vmap(check_spread, in_axes=[0, None, None]),
            in_axes=[None, 0, None],
        ),
        in_axes=[None, None, 0],
    )(jnp.arange(n * n, dtype=jnp.uint32), 1 + jnp.arange(num_spread_patterns), jnp.arange(4, dtype=jnp.uint8))

    spreads = jnp.concatenate(spreads, axis=1).transpose().reshape((num_spread_patterns, n, n))
    mask = mask.at[2:].set(spreads)

    return mask
