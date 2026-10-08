#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export PYTHONNOUSERSITE=1
export APPWORLD_DATASET_NAME=${APPWORLD_DATASET_NAME:-appworld_official_test_normal}
export APPWORLD_MAX_TURNS=${APPWORLD_MAX_TURNS:-20}
export APPWORLD_ENV_CONCURRENCY=${APPWORLD_ENV_CONCURRENCY:-8}
if [[ -z "${DRY_RUN:-}" && -z "${APPWORLD_REMOTE_ENVIRONMENT_URLS:-}" && -z "${APPWORLD_REMOTE_ENVIRONMENT_URL:-}" ]]; then
    echo "Set APPWORLD_REMOTE_ENVIRONMENT_URLS to running AppWorld servers." >&2
    exit 1
fi
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/sc_grpo_ray_${USER}}

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_batch_size=8
    data.val_batch_size=16
    data.max_prompt_length=8192
    data.max_response_length=12288
    actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen3-8B}"
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode="async"
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.val_kwargs.n=8
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24000
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    algorithm.kl_ctrl.kl_coef=0
    rllm.mask_truncated_samples=False
    rllm.disable_thinking=True
    rllm.agent.max_steps=${APPWORLD_MAX_TURNS}
    rllm.agent.trajectory_timeout=1200
    rllm.stepwise_advantage.enable=False
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=sc-grpo
    trainer.log_val_generations=1
    trainer.val_before_train=True
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}
    trainer.nnodes=${N_NODES:-1}
    trainer.save_freq=100
    trainer.test_freq=20
    trainer.default_hdfs_dir=null
    trainer.total_epochs=${TOTAL_TRAINING_STEPS:-110}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-110}
    +rllm.agent.engine_args.max_concurrency=${APPWORLD_ENV_CONCURRENCY}
    +ray_init._temp_dir=${RAY_TMPDIR}
    +ray_init.include_dashboard=False
)

SC_GRPO_ARGS=(
    rllm.sc_grpo.enable=True
    rllm.sc_grpo.positive_reward_threshold=1.0
    rllm.sc_grpo.topk=64
    rllm.sc_grpo.explore_coeff=0.1
    rllm.sc_grpo.kl_weight_mode=mask
    rllm.sc_grpo.kl_c_quantile=0.75
    rllm.sc_grpo.kl_c_floor=1e-4
    rllm.sc_grpo.diversity_ref_filter_enable=False
    trainer.experiment_name=sc-grpo-appworld
)
COMMAND=("${PYTHON_BIN:-python3}" -m examples.appworld.train_appworld_agent "${COMMON_ARGS[@]}" "${SC_GRPO_ARGS[@]}" "$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    exec "${COMMAND[@]}"
fi
