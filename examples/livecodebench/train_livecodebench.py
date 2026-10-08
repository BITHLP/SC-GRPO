import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError

from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import code_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

try:
    from examples.livecodebench.prepare_livecodebench_data import prepare_livecodebench_data
except ModuleNotFoundError:
    from prepare_livecodebench_data import prepare_livecodebench_data


DEFAULT_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _parse_logger(value: str):
    loggers = [item.strip() for item in value.split(",") if item.strip()]
    return loggers or ["wandb"]


def _parse_nullable_str(value: str):
    lowered = value.strip().lower()
    if lowered in {"", "null", "none"}:
        return None
    return value


def _load_with_env(env_name: str, default_value, parser):
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default_value
    return parser(raw_value)


def _cli_overridden_keys() -> set:
    overrides = HydraConfig.get().overrides.task
    keys = set()
    for item in overrides:
        if "=" not in item:
            continue
        key = item.split("=", 1)[0].lstrip("+~")
        if key:
            keys.add(key)
    return keys


def _apply_training_defaults(config) -> None:
    cli_keys = _cli_overridden_keys()
    defaults = [
        ("algorithm.adv_estimator", "LCB_ADV_ESTIMATOR", str, "grpo"),
        ("data.train_batch_size", "LCB_TRAIN_BATCH_SIZE", int, 128),
        ("data.val_batch_size", "LCB_VAL_BATCH_SIZE", int, 256),
        ("data.max_prompt_length", "LCB_MAX_PROMPT_LENGTH", int, 2048),
        ("data.max_response_length", "LCB_MAX_RESPONSE_LENGTH", int, 16384),
        ("actor_rollout_ref.model.path", "LCB_MODEL_PATH", str, DEFAULT_MODEL_PATH),
        ("actor_rollout_ref.hybrid_engine", "LCB_HYBRID_ENGINE", _parse_bool, True),
        ("actor_rollout_ref.rollout.name", "LCB_ROLLOUT_NAME", str, "vllm"),
        ("actor_rollout_ref.rollout.mode", "LCB_ROLLOUT_MODE", str, "async"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", "LCB_ROLLOUT_TP_SIZE", int, 1),
        ("actor_rollout_ref.rollout.temperature", "LCB_ROLLOUT_TEMPERATURE", float, 0.6),
        ("actor_rollout_ref.rollout.top_p", "LCB_ROLLOUT_TOP_P", float, 0.95),
        ("actor_rollout_ref.rollout.gpu_memory_utilization", "LCB_ROLLOUT_GPU_MEMORY_UTILIZATION", float, 0.8),
        ("actor_rollout_ref.rollout.n", "LCB_ROLLOUT_N", int, 8),
        ("actor_rollout_ref.rollout.val_kwargs.n", "LCB_VAL_ROLLOUT_N", int, 2),
        ("actor_rollout_ref.rollout.val_kwargs.temperature", "LCB_VAL_TEMPERATURE", float, 0.6),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", "LCB_VAL_TOP_P", float, 0.95),
        ("actor_rollout_ref.actor.optim.lr", "LCB_ACTOR_LR", float, 1e-6),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", "LCB_PPO_MINI_BATCH_SIZE", int, 64),
        ("actor_rollout_ref.actor.use_dynamic_bsz", "LCB_USE_DYNAMIC_BSZ", _parse_bool, True),
        ("trainer.project_name", "LCB_PROJECT_NAME", str, "rllm-livecodebench"),
        ("trainer.experiment_name", "LCB_EXPERIMENT_NAME", str, "livecodebench-agent"),
        ("trainer.logger", "LCB_TRAINER_LOGGER", _parse_logger, ["wandb"]),
        ("trainer.log_val_generations", "LCB_LOG_VAL_GENERATIONS", int, 1),
        ("trainer.val_before_train", "LCB_VAL_BEFORE_TRAIN", _parse_bool, True),
        ("trainer.n_gpus_per_node", "LCB_N_GPUS_PER_NODE", int, 8),
        ("trainer.nnodes", "LCB_N_NODES", int, 1),
        ("trainer.save_freq", "LCB_SAVE_FREQ", int, 10),
        ("trainer.test_freq", "LCB_TEST_FREQ", int, 10),
        ("trainer.total_epochs", "LCB_TOTAL_EPOCHS", int, 100),
        ("trainer.default_hdfs_dir", "LCB_DEFAULT_HDFS_DIR", _parse_nullable_str, None),
        ("rllm.agent.max_steps", "LCB_AGENT_MAX_STEPS", int, 1),
        ("rllm.stepwise_advantage.enable", "LCB_STEPWISE_ADV_ENABLE", _parse_bool, False),
        ("rllm.sc_grpo.enable", "LCB_SC_GRPO_ENABLE", _parse_bool, False),
        ("rllm.sc_grpo.positive_reward_threshold", "LCB_SC_GRPO_POSITIVE_REWARD_THRESHOLD", float, 1.0),
        ("rllm.sc_grpo.topk", "LCB_SC_GRPO_TOPK", int, 64),
        ("rllm.sc_grpo.max_reference_chars", "LCB_SC_GRPO_MAX_REFERENCE_CHARS", int, 12000),
    ]

    for key, env_name, parser, default_value in defaults:
        if key in cli_keys:
            continue
        value = _load_with_env(env_name, default_value, parser)
        try:
            OmegaConf.update(config, key, value, merge=True)
        except ConfigAttributeError:
            with open_dict(config):
                OmegaConf.update(config, key, value, merge=True)


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    dataset_name = os.getenv("LCB_DATASET_NAME", "livecodebench_v6")
    force_prepare = _parse_bool(os.getenv("LCB_FORCE_PREPARE", "false"))
    start_date = os.getenv("LCB_START_DATE", "2025-02")
    end_date = os.getenv("LCB_END_DATE", "2025-06")
    public_test_ratio = float(os.getenv("LCB_PUBLIC_TEST_RATIO", "0.5"))
    seed = int(os.getenv("LCB_SEED", "42"))

    _apply_training_defaults(config)

    train_dataset = None if force_prepare else DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = None if force_prepare else DatasetRegistry.load_dataset(dataset_name, "test")

    if train_dataset is None or val_dataset is None:
        if force_prepare:
            print("Preparing LiveCodeBench v6 dataset (LCB_FORCE_PREPARE=true)...")
        else:
            print(f"Dataset '{dataset_name}' not found. Preparing from HuggingFace...")
        prepare_livecodebench_data(
            start_date=start_date,
            end_date=end_date,
            public_test_ratio=public_test_ratio,
            seed=seed,
            dataset_name=dataset_name,
        )
        train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
        val_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}'.")

    trainer = AgentTrainer(
        agent_class=CompetitionCodingAgent,
        agent_args={},
        env_args={"reward_fn": code_reward_fn},
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
