from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp


def supported_size(n: int) -> bool:
    return 3 <= n <= 8


def possible_moves(n: int) -> int:
    assert supported_size(n)
    return [126, 480, 1_575, 4_572, 12_495, 32_704][n - 3]


class Action(NamedTuple):
    square: chex.Array = jnp.uint8(0)
    pattern: chex.Array = jnp.uint8(0)


def spread_patterns(n: int) -> int:
    return (1 << n) - 2


def action_index(n: int, action: Action) -> chex.Array:
    """
    Get an index for an action, assuming a policy output shape
    where each channel corresponds to a different move type,
    and the location in each channel corresponds to the starting
    square of the move.
    """
    assert supported_size(n)
    square, pattern = action
    placement = pattern == 0
    row = square & jnp.uint8(0b111)
    col = jnp.right_shift(square & jnp.uint8(0b111_000), 3)
    direction_or_piece = jnp.right_shift(square & jnp.uint8(0b11_000_000), 6)

    def if_placement() -> chex.Array:
        assert 1 <= direction_or_piece <= 3
        # 0b01 = FLAT
        # 0b10 = WALL
        # 0b11 = CAP
        return direction_or_piece

    def if_spread() -> chex.Array:
        all_patterns = spread_patterns(n)
        direction_offset = direction_or_piece * all_patterns
        # Valid patterns start at 1
        pattern_offset = pattern - 1
        return 3 + direction_offset + pattern_offset

    channel = jax.lax.cond(
        placement,
        if_placement,
        if_spread,
    )
    return channel * n * n + row * n + col
