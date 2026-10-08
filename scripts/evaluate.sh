#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK=${1:?Usage: evaluate.sh math|livecodebench|appworld|webshop MODEL_PATH [Hydra overrides]}
MODEL_PATH=${2:?Provide a Hugging Face model directory or model ID}
shift 2
case "$TASK" in
    math) ENTRY=examples/dapo_math/train_dapo_math.sh ;;
    livecodebench) ENTRY=examples/livecodebench/train_livecodebench.sh ;;
    appworld) ENTRY=examples/appworld/train_appworld_agent.sh ;;
    webshop) ENTRY=examples/webshop/train_webshop_agent.sh ;;
    *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
exec bash "$REPO_ROOT/$ENTRY" "trainer.val_only=True" trainer.val_before_train=True     "actor_rollout_ref.model.path=$MODEL_PATH" actor_rollout_ref.rollout.val_kwargs.n=8 "$@"
