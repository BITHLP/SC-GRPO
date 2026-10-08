"""Check release syntax, launch commands, and optional Hydra composition without GPUs."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib
import warnings

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    "math": ("examples/dapo_math/train_dapo_math.sh", 500),
    "livecodebench": ("examples/livecodebench/train_livecodebench.sh", 320),
    "appworld": ("examples/appworld/train_appworld_agent.sh", 110),
    "webshop": ("examples/webshop/train_webshop_agent.sh", 300),
}


def compose_config(tokens):
    """Load the real YAML config tree without importing CUDA/Ray backend code."""
    import types
    import importlib.util
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    package_roots = {
        "rllm": ROOT / "rllm",
        "rllm.trainer": ROOT / "rllm/trainer",
        "rllm.trainer.config": ROOT / "rllm/trainer/config",
        "verl": ROOT / "verl/verl",
        "verl.trainer": ROOT / "verl/verl/trainer",
        "verl.trainer.config": ROOT / "verl/verl/trainer/config",
    }
    for name, path in package_roots.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__file__ = str(path / "__init__.py")
        package.__spec__ = importlib.util.spec_from_file_location(name, package.__file__, submodule_search_locations=[str(path)])
        package.__loader__ = package.__spec__.loader
        sys.modules[name] = package
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "rllm/trainer/config"), version_base=None):
        config = compose(config_name="agent_ppo_trainer", overrides=tokens[3:])
    assert config.rllm.sc_grpo.enable
    assert config.rllm.sc_grpo.kl_weight_mode == "mask"
    assert config.actor_rollout_ref.rollout.val_kwargs.n == 8
    return config


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
    for path in ROOT.rglob("*.toml"):
        tomllib.loads(path.read_text())
    for path in list((ROOT / "examples").rglob("*.sh")) + list((ROOT / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(path)], check=True)
    print("PASS: Python, TOML and shell syntax")

    hydra_available = False
    try:
        import hydra  # noqa: F401
        hydra_available = True
    except ImportError:
        pass
    env = dict(os.environ, DRY_RUN="1", TOTAL_TRAINING_STEPS="", MODEL_PATH="/tmp/model with spaces")
    env.pop("TOTAL_TRAINING_STEPS")
    for task, (entry, steps) in ENTRIES.items():
        output = subprocess.check_output(["bash", str(ROOT / entry)], env=env, text=True)
        tokens = shlex.split(output)
        values = dict(token.split("=", 1) for token in tokens[3:])
        assert values["rllm.sc_grpo.enable"] == "True"
        assert values["trainer.total_training_steps"] == str(steps)
        assert values["actor_rollout_ref.rollout.val_kwargs.n"] == "8"
        assert values["actor_rollout_ref.model.path"] == "/tmp/model with spaces"
        assert values["rllm.disable_thinking"] == "True"
        assert values["algorithm.adv_estimator"] == "grpo"
        if hydra_available:
            compose_config(tokens)
        eval_output = subprocess.check_output(
            ["bash", str(ROOT / "scripts/evaluate.sh"), task, "/tmp/exported model"],
            env=env, text=True,
        )
        eval_tokens = shlex.split(eval_output)
        assert "trainer.val_only=True" in eval_tokens
        if hydra_available:
            assert compose_config(eval_tokens).trainer.val_only
        print(f"PASS: {task} training/evaluation launchers" + (" and Hydra composition" if hydra_available else ""))

    # Scan text files; report paths only, never matching credential values.
    bad_paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".yaml", ".yml", ".md", ".json", ".toml"}:
            continue
        if path == Path(__file__).resolve():
            continue
        text = path.read_text()
        if re.search(r"dolphinfs|hadoop-kg|git\.sankuai|export SWANLAB_API_KEY=[A-Za-z0-9]{12,}", text):
            bad_paths.append(str(path.relative_to(ROOT)))
    if bad_paths:
        raise RuntimeError("Private configuration found in: " + ", ".join(bad_paths))
    print("PASS: private-path and hardcoded-key scan")
    if not hydra_available:
        print("Hydra composition skipped: install hydra-core to enable it.")


if __name__ == "__main__":
    main()
