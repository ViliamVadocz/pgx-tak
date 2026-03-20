import jax.numpy as jnp
from jax.experimental import checkify

BITBOARD_DTYPE = jnp.uint64
BITBOARD_IINFO = jnp.iinfo(BITBOARD_DTYPE)

STACK_DTYPE = jnp.uint64
STACK_IINFO = jnp.iinfo(STACK_DTYPE)
MAX_STACK_SIZE = STACK_IINFO.bits - 1

MIN_SIZE = 3
MAX_SIZE = 8
DEFAULT_SIZE = 6


def check_size(size: int) -> None:
    checkify.check(size >= MIN_SIZE, "board size too small")
    checkify.check(size <= MAX_SIZE, "board size too big")


def starting_reserves(size: int) -> tuple[int, int]:
    check_size(size)
    stones = [10, 15, 21, 30, 40, 50]
    caps = [0, 0, 1, 1, 2, 2]
    i = size - 3
    return stones[i], caps[i]


DEFAULT_STONES, DEFAULT_CAPS = starting_reserves(DEFAULT_SIZE)
