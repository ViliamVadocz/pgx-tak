from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify  # https://docs.jax.dev/en/latest/debugging/checkify_guide.html

import bitboard
import zobrist
from consts import BITBOARD_DTYPE, DEFAULT_CAPS, DEFAULT_SIZE, DEFAULT_STONES, MAX_STACK_SIZE, check_size


class State(NamedTuple):
    size: int = DEFAULT_SIZE
    starting_stones: int = DEFAULT_STONES
    starting_caps: int = DEFAULT_CAPS

    white_to_move: jax.Array = jnp.bool(True)
    stacks: jax.Array = jnp.ones(DEFAULT_SIZE * DEFAULT_SIZE, dtype=jnp.uint64)
    zobrist_hash: jax.Array = zobrist.BOARD_SIZE[DEFAULT_SIZE] ^ zobrist.WHITE_TO_MOVE

    # bitboards
    white_bb: jax.Array = BITBOARD_DTYPE(0)
    black_bb: jax.Array = BITBOARD_DTYPE(0)
    road_bb: jax.Array = BITBOARD_DTYPE(0)
    noble_bb: jax.Array = BITBOARD_DTYPE(0)

    # reserves
    white_stones: jax.Array = jnp.uint8(DEFAULT_STONES)
    black_stones: jax.Array = jnp.uint8(DEFAULT_STONES)
    white_caps: jax.Array = jnp.uint2(DEFAULT_CAPS)
    black_caps: jax.Array = jnp.uint2(DEFAULT_CAPS)


def check_invariants(state: State) -> None:
    check_size(state.size)

    # Calculate number of pieces on the board from stacks
    stack_heights = MAX_STACK_SIZE - jax.lax.clz(state.stacks)
    white_pieces = jax.lax.population_count(state.stacks) - 1
    black_pieces = stack_heights - white_pieces
    white_pieces = jnp.sum(white_pieces)
    black_pieces = jnp.sum(black_pieces)

    # Figure out how many pieces are capstones
    caps = state.road_bb & state.noble_bb
    white_caps = jax.lax.population_count(caps & state.white_bb)
    black_caps = jax.lax.population_count(caps & state.black_bb)
    white_flats = white_pieces - white_caps
    black_flats = black_pieces - black_caps

    # Check reserves consistent with stacks
    checkify.check(state.white_stones == state.starting_stones - white_flats, "incorrect white stone reserves")
    checkify.check(state.black_stones == state.starting_stones - black_flats, "incorrect black stone reserves")
    checkify.check(
        state.white_caps.astype(jnp.uint8) == state.starting_caps - white_caps, "incorrect white cap reserves"
    )
    checkify.check(
        state.black_caps.astype(jnp.uint8) == state.starting_caps - black_caps, "incorrect black cap reserves"
    )

    walls = state.noble_bb & ~state.road_bb
    flats = ~state.noble_bb & state.road_bb
    not_empty = stack_heights > 0
    black = (state.stacks & 0b1) == 0
    white = not_empty & ~black

    # Check bitboards consistent with reserves and stacks
    checkify.check(state.white_bb == bitboard.mask_to_bb(white), "incorrect white bitboard")
    checkify.check(state.black_bb == bitboard.mask_to_bb(black), "incorrect black bitboard")
    checkify.check(state.white_bb & state.black_bb == 0, "white and black bitboard overlap")

    any_color = state.white_bb | state.black_bb
    any_piece = state.road_bb | state.noble_bb
    checkify.check(any_color == any_piece, "inconsistent color and piece bitboards")

    wall_count = jax.lax.population_count(walls)
    flat_count = jax.lax.population_count(flats)
    checkify.check(jax.lax.population_count(caps) == white_caps + black_caps, "incorrect cap reserves")
    checkify.check(wall_count + flat_count == white_flats + black_flats, "incorrect stone reserves")

    checkify.check(state.zobrist_hash == zobrist.get_hash(state), "incorrect hash")


def test_invariants() -> None:
    state = State()
    err: checkify.Error
    err, _ = jax.jit(checkify.checkify(check_invariants, errors=checkify.all_checks))(state)
    err.throw()
