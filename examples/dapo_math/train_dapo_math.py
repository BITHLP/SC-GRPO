import os

import hydra

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


def _load_required_dataset(name: str, split: str):
    dataset = DatasetRegistry.load_dataset(name, split)
    if dataset is None:
        raise RuntimeError(
            f"DatasetRegistry entry {name!r}/{split!r} was not found. "
            "Run `python3 -m examples.dapo_math.prepare_math_data` first."
        )
    return dataset


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset_name = os.environ.get("RLLM_DAPO_TRAIN_DATASET", "dapo_math_4k")
    val_dataset_name = os.environ.get("RLLM_DAPO_VAL_DATASET", "aime24_25")

    train_dataset = _load_required_dataset(train_dataset_name, "train")
    val_dataset = _load_required_dataset(val_dataset_name, "test")

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={},
        env_args={"reward_fn": math_reward_fn},
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
