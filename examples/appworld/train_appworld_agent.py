import os

import hydra

from rllm.agents.appworld_react_agents import AppWorldReactAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.appworld.appworld_env import AppWorldEnv
from rllm.environments.appworld.signal_patch import apply_signal_patch
from rllm.trainer.agent_trainer import AgentTrainer

try:
    from examples.appworld.prepare_appworld_data import prepare_appworld_data
except ModuleNotFoundError:
    from prepare_appworld_data import prepare_appworld_data


apply_signal_patch(verbose=False)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip().lower() in {"", "none", "null", "-1"}:
        return None
    return int(value)


def _remote_appworld_enabled() -> bool:
    return bool(
        os.getenv("APPWORLD_REMOTE_ENVIRONMENT_URLS", "").strip()
        or os.getenv("APPWORLD_REMOTE_ENVIRONMENT_URL", "").strip()
    )


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    dataset_name = os.getenv("APPWORLD_DATASET_NAME", "appworld_official_dev")
    force_prepare = _parse_bool(os.getenv("APPWORLD_FORCE_PREPARE", "false"))
    train_split = os.getenv("APPWORLD_TRAIN_SPLIT", "train")
    val_split = os.getenv("APPWORLD_VAL_SPLIT", "dev")
    train_size = _parse_optional_int(os.getenv("APPWORLD_TRAIN_SIZE"))
    val_size = _parse_optional_int(os.getenv("APPWORLD_VAL_SIZE", "none"))
    max_turns = int(os.getenv("APPWORLD_MAX_TURNS", "20"))

    train_dataset = None if force_prepare else DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = None if force_prepare else DatasetRegistry.load_dataset(dataset_name, "test")

    if train_dataset is None or val_dataset is None:
        if _remote_appworld_enabled() and not force_prepare:
            raise RuntimeError(
                f"Dataset {dataset_name!r} was not found. Prepare it once in the AppWorld/Pydantic-v1 "
                "environment with `python -m examples.appworld.prepare_appworld_data`, then rerun training "
                "from the verl/vLLM environment."
            )
        if force_prepare:
            print(f"Preparing AppWorld dataset {dataset_name} (APPWORLD_FORCE_PREPARE=true)...")
        else:
            print(f"Dataset {dataset_name} not found. Preparing AppWorld official tasks...")
        prepare_appworld_data(
            train_split=train_split,
            val_split=val_split,
            train_size=train_size,
            val_size=val_size,
            dataset_name=dataset_name,
        )
        train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
        val_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError(f"Failed to load dataset {dataset_name}.")

    trainer = AgentTrainer(
        agent_class=AppWorldReactAgent,
        agent_args={},
        env_class=AppWorldEnv,
        env_args={"max_turns": max_turns},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
