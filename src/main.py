import chex
import jax.numpy as jnp
import pgx
from pgx._src.struct import dataclass

type Array = chex.Array

ENV_ID = "tak"


@dataclass
class TakState(pgx.State):
    observation: Array = jnp.zeros(0, dtype=jnp.bool)  # depends on size
    current_player: Array = jnp.uint32(0)
    rewards: Array = jnp.zeros(2, dtype=jnp.float32)
    terminated: Array = jnp.bool(False)
    truncated: Array = jnp.bool(False)
    legal_action_mask: Array = jnp.ones(0, dtype=jnp.bool)  # depends on size
    _step_count: Array = jnp.uint32(0)

    @property
    def env_id(self) -> pgx.EnvId:
        """Environment id (e.g. "go_19x19")"""
        return ENV_ID  # type: ignore


class Tak(pgx.Env):
    """TODO"""

    def __init__(self, size: int = 8, half_komi: int = 4) -> None:
        """
        :param size: The largest number that can be picked.
        """
        super().__init__()
        self.size = size
        self.half_komi = half_komi

    def _init(self, key: chex.PRNGKey) -> TakState:
        del key
        return TakState()  # TODO

    def _step(self, state: TakState, action: Array, key: chex.PRNGKey) -> TakState:
        del key
        assert isinstance(state, TakState)
        del action  # TODO
        return state  # TODO

    def _observe(self, state: pgx.State, player_id: Array) -> Array:
        del player_id
        assert isinstance(state, TakState)
        return state.observation

    @property
    def id(self) -> pgx.EnvId:
        """Environment id."""
        return ENV_ID  # type: ignore

    @property
    def version(self) -> str:
        """Environment version. Updated when behavior, parameter, or API is changed.
        Refactoring or speeding up without any expected behavior changes will NOT update the version number.
        """
        return "0.0.1"

    @property
    def num_players(self) -> int:
        """Number of players (e.g., 2 in Tic-tac-toe)"""
        return 2
