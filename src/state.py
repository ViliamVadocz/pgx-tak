from typing import NamedTuple

import chex
import jax.numpy as jnp

n = 6
STARTPOS_HASH = 0  # TODO
MAX_STACK_SIZE = 63  # number of bits in stack element - 1
starting_stones = 30
starting_capstones = 1


class State(NamedTuple):
    size: int = n

    white_to_move: chex.Array = jnp.bool(True)
    stacks: chex.Array = jnp.ones(n * n, dtype=jnp.uint64)
    zobrist_hash: chex.Array = jnp.uint64(STARTPOS_HASH)

    # bitboards
    white_bb: chex.Array = jnp.uint64(0)
    black_bb: chex.Array = jnp.uint64(0)
    road_bb: chex.Array = jnp.uint64(0)
    noble_bb: chex.Array = jnp.uint64(0)

    # reserves
    white_stones: chex.Array = jnp.uint8(starting_stones)
    black_stones: chex.Array = jnp.uint8(starting_stones)
    white_capstones: chex.Array = jnp.uint2(starting_capstones)
    black_capstones: chex.Array = jnp.uint2(starting_capstones)


def get_hash(state: State) -> chex.Array:
    return jnp.uint64(STARTPOS_HASH)  # TODO


# TODO: Assertions in JAX?
def check_invariants(state: State) -> None:
    # Calculate number of pieces on the board from stacks
    stack_heights = MAX_STACK_SIZE - jnp.leading_zeros(state.stacks)
    white_pieces = jnp.population_count(state.stacks) - 1
    black_pieces = stack_heights - white_flats
    white_pieces = jnp.sum(white_pieces)
    black_pieces = jnp.sum(black_pieces)

    # Figure out how many pieces are capstones
    caps = jnp.bitwise_and(state.road_bb, state.noble_bb)
    white_caps = jnp.population_count(jnp.bitwise_and(caps, state.white_bb))
    black_caps = jnp.population_count(jnp.bitwise_and(caps, state.black_bb))
    white_flats = white_pieces - white_caps
    black_flats = black_pieces - black_caps

    # Check reserves
    assert state.white_stones == starting_stones - white_flats
    assert state.black_stones == starting_stones - black_flats
    assert state.white_capstones == starting_capstones - white_caps
    assert state.black_capstones == starting_capstones - black_caps

    walls = jnp.bitwise_and(state.noble_bb, jnp.bitwise_not(state.road_bb))
    flats = jnp.bitwise_and(jnp.bitwise_not(state.noble_bb), state.road_bb)
    not_empty = stack_heights > 0
    black = jnp.bitwise_and(state.stacks, 0b1) == 0
    white = not_empty and jnp.logical_not(black)

    # Check bitboards
    assert state.white_bb == white
    assert state.black_bb == black
    assert jnp.bitwise_and(state.white_bb, state.black_bb) == 0
    assert jnp.bitwise_or(state.white_bb, state.black_bb) == jnp.bitwise_or(state.road_bb, state.noble_bb)
    assert jnp.population_count(caps) == white_caps + black_caps
    assert jnp.population_count(walls) + jnp.population_count(flats) == white_flats + black_flats

    assert state.hash == get_hash(state)
