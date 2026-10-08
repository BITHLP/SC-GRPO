import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError

from rllm.data import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

try:
    from examples.webshop.prepare_webshop_data import prepare_webshop_data
    from examples.webshop.webshop_agent import WebShopAgent
    from examples.webshop.webshop_env import WebShopTextEnv
except ModuleNotFoundError:
    from prepare_webshop_data import prepare_webshop_data
    from webshop_agent import WebShopAgent
    from webshop_env import WebShopTextEnv


DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _parse_logger(value: str):
    loggers = [item.strip() for item in value.split(",") if item.strip()]
    return loggers or ["swanlab"]


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


def _has_webshop_metadata(dataset) -> bool:
    if dataset is None or len(dataset) == 0:
        return False
    sample = dataset[0]
    return isinstance(sample, dict) and sample.get("data_source") == "webshop"


def _has_expected_size(dataset, expected_size: int) -> bool:
    return dataset is not None and len(dataset) == expected_size


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


def _apply_training_defaults(config, env_max_steps: int) -> None:
    cli_keys = _cli_overridden_keys()
    defaults = [
        ("algorithm.adv_estimator", "WEBSHOP_ADV_ESTIMATOR", str, "grpo"),
        ("data.train_batch_size", "WEBSHOP_TRAIN_BATCH_SIZE", int, 4),
        ("data.val_batch_size", "WEBSHOP_VAL_BATCH_SIZE", int, 32),
        ("data.max_prompt_length", "WEBSHOP_MAX_PROMPT_LENGTH", int, 8192),
        ("data.max_response_length", "WEBSHOP_MAX_RESPONSE_LENGTH", int, 8192),
        ("actor_rollout_ref.model.path", "WEBSHOP_MODEL_PATH", str, DEFAULT_MODEL_PATH),
        ("actor_rollout_ref.hybrid_engine", "WEBSHOP_HYBRID_ENGINE", _parse_bool, True),
        ("actor_rollout_ref.rollout.name", "WEBSHOP_ROLLOUT_NAME", str, "vllm"),
        ("actor_rollout_ref.rollout.mode", "WEBSHOP_ROLLOUT_MODE", str, "async"),
        (
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            "WEBSHOP_ROLLOUT_TP_SIZE",
            int,
            1,
        ),
        ("actor_rollout_ref.rollout.temperature", "WEBSHOP_ROLLOUT_TEMPERATURE", float, 0.7),
        (
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "WEBSHOP_ROLLOUT_GPU_MEMORY_UTILIZATION",
            float,
            0.65,
        ),
        ("actor_rollout_ref.rollout.n", "WEBSHOP_ROLLOUT_N", int, 8),
        ("actor_rollout_ref.rollout.val_kwargs.n", "WEBSHOP_VAL_ROLLOUT_N", int, 1),
        (
            "actor_rollout_ref.rollout.val_kwargs.temperature",
            "WEBSHOP_VAL_TEMPERATURE",
            float,
            0.6,
        ),
        ("actor_rollout_ref.rollout.val_kwargs.top_p", "WEBSHOP_VAL_TOP_P", float, 0.95),
        ("actor_rollout_ref.actor.optim.lr", "WEBSHOP_ACTOR_LR", float, 1e-6),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", "WEBSHOP_PPO_MINI_BATCH_SIZE", int, 8),
        ("actor_rollout_ref.actor.use_dynamic_bsz", "WEBSHOP_USE_DYNAMIC_BSZ", _parse_bool, True),
        ("trainer.project_name", "WEBSHOP_PROJECT_NAME", str, "rllm-agent"),
        ("trainer.experiment_name", "WEBSHOP_EXPERIMENT_NAME", str, "webshop-agent-remote"),
        ("trainer.logger", "WEBSHOP_TRAINER_LOGGER", _parse_logger, ["swanlab"]),
        ("trainer.log_val_generations", "WEBSHOP_LOG_VAL_GENERATIONS", int, 1),
        ("trainer.val_before_train", "WEBSHOP_VAL_BEFORE_TRAIN", _parse_bool, False),
        ("trainer.n_gpus_per_node", "WEBSHOP_N_GPUS_PER_NODE", int, 8),
        ("trainer.nnodes", "WEBSHOP_N_NODES", int, 1),
        ("trainer.save_freq", "WEBSHOP_SAVE_FREQ", int, 20),
        ("trainer.test_freq", "WEBSHOP_TEST_FREQ", int, 10),
        ("trainer.default_hdfs_dir", "WEBSHOP_DEFAULT_HDFS_DIR", _parse_nullable_str, None),
        ("rllm.agent.max_steps", "WEBSHOP_AGENT_MAX_STEPS", int, env_max_steps),
        ("rllm.agent.env_init_max_workers", "WEBSHOP_ENV_INIT_MAX_WORKERS", int, 8),
        ("rllm.agent.trajectory_timeout", "WEBSHOP_TRAJECTORY_TIMEOUT", int, 600),
        ("rllm.agent.engine_args.max_concurrency", "WEBSHOP_ENGINE_MAX_CONCURRENCY", int, 8),
        ("rllm.stepwise_advantage.enable", "WEBSHOP_STEPWISE_ADV_ENABLE", _parse_bool, False),
        ("rllm.sc_grpo.enable", "WEBSHOP_SC_GRPO_ENABLE", _parse_bool, False),
        (
            "rllm.sc_grpo.positive_reward_threshold",
            "WEBSHOP_SC_GRPO_POSITIVE_REWARD_THRESHOLD",
            float,
            1.0,
        ),
        ("rllm.sc_grpo.topk", "WEBSHOP_SC_GRPO_TOPK", int, 64),
        ("rllm.sc_grpo.max_reference_chars", "WEBSHOP_SC_GRPO_MAX_REFERENCE_CHARS", int, 12000),
        ("trainer.total_epochs", "WEBSHOP_TOTAL_EPOCHS", int, 3),
    ]

    for key, env_name, parser, default_value in defaults:
        if key in cli_keys:
            continue
        value = _load_with_env(env_name, default_value, parser)
        try:
            OmegaConf.update(config, key, value, merge=True)
        except ConfigAttributeError:
            # Some trainer configs are strict (struct mode) and may not expose all optional keys.
            with open_dict(config):
                OmegaConf.update(config, key, value, merge=True)


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    split_mode = os.getenv("WEBSHOP_SPLIT_MODE", "official").strip().lower()
    if split_mode not in {"official", "random"}:
        raise ValueError(f"Unsupported WEBSHOP_SPLIT_MODE={split_mode!r}. Expected 'official' or 'random'.")

    default_dataset_name = "webshop_text_official" if split_mode == "official" else "webshop_text"
    dataset_name = os.getenv("WEBSHOP_DATASET_NAME", default_dataset_name)
    force_prepare = _parse_bool(os.getenv("WEBSHOP_FORCE_PREPARE", "false"))
    train_size_raw = os.getenv("WEBSHOP_TRAIN_SIZE", "2400")
    train_size = int(train_size_raw) if train_size_raw else None
    test_size = int(os.getenv("WEBSHOP_TEST_SIZE", "100"))

    num_products = int(os.getenv("WEBSHOP_NUM_PRODUCTS", "1000"))
    human_goals = int(os.getenv("WEBSHOP_HUMAN_GOALS", "0"))
    env_max_steps = int(os.getenv("WEBSHOP_ENV_MAX_STEPS", "15"))
    observation_mode = os.getenv("WEBSHOP_OBSERVATION_MODE", "text_rich")
    webshop_root = os.getenv("WEBSHOP_ROOT")

    _apply_training_defaults(config, env_max_steps=env_max_steps)

    train_dataset = None if force_prepare else DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = None if force_prepare else DatasetRegistry.load_dataset(dataset_name, "test")

    stale_metadata = False
    if train_dataset is not None and not _has_webshop_metadata(train_dataset):
        stale_metadata = True
        train_dataset = None
    if val_dataset is not None and not _has_webshop_metadata(val_dataset):
        stale_metadata = True
        val_dataset = None
    if val_dataset is not None and not _has_expected_size(val_dataset, test_size):
        stale_metadata = True
        val_dataset = None

    if train_dataset is None or val_dataset is None:
        if force_prepare:
            print(f"Preparing WebShop dataset {dataset_name} from local backend (WEBSHOP_FORCE_PREPARE=true)...")
        elif stale_metadata:
            print(f"Dataset {dataset_name} metadata or split size is stale. Re-preparing from local backend...")
        else:
            print(f"Dataset {dataset_name} not found. Preparing WebShop dataset from local backend...")
        prepare_webshop_data(
            train_size=train_size,
            test_size=test_size,
            split_mode=split_mode,
            num_products=num_products,
            human_goals=human_goals,
            observation_mode=observation_mode,
            webshop_root=webshop_root,
            dataset_name=dataset_name,
        )
        train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
        val_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    if train_dataset is None or val_dataset is None:
        raise RuntimeError(f"Failed to load prepared dataset {dataset_name}.")

    env_args = {
        "observation_mode": observation_mode,
        "num_products": num_products,
        "human_goals": human_goals,
        "max_steps": env_max_steps,
        "session_prefix": "rllm_",
        "webshop_root": webshop_root,
    }

    agent_args = {
        "use_full_conversation": True,
        "strip_thinking": True,
        "include_available_actions": True,
    }

    trainer = AgentTrainer(
        agent_class=WebShopAgent,
        env_class=WebShopTextEnv,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
