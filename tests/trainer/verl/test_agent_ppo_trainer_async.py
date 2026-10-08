import pytest

from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer


class _FailingTrajectoryEngine:
    async def trajectory_generator(self, timing_raw=None, mode="Token", meta_info=None):
        raise RuntimeError("trajectory failed")
        yield  # pragma: no cover


def test_generate_agent_trajectories_async_propagates_background_exceptions():
    trainer = object.__new__(AgentPPOTrainer)
    trainer.agent_execution_engine = _FailingTrajectoryEngine()

    generator = trainer.generate_agent_trajectories_async()

    with pytest.raises(RuntimeError, match="trajectory failed"):
        next(generator)
