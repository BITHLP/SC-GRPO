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
export WEBSHOP_TRAIN_SIZE=${WEBSHOP_TRAIN_SIZE:-2400}
export WEBSHOP_TEST_SIZE=${WEBSHOP_TEST_SIZE:-100}
export WEBSHOP_NUM_PRODUCTS=${WEBSHOP_NUM_PRODUCTS:-1000}
export WEBSHOP_DATASET_NAME=${WEBSHOP_DATASET_NAME:-webshop_text_official}
if [[ -z "${DRY_RUN:-}" && -z "${WEBSHOP_ROOT:-}" ]]; then
    echo "Set WEBSHOP_ROOT to your WebShop checkout." >&2
    exit 1
fi

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_batch_size=8
    data.val_batch_size=64
    data.max_prompt_length=4096
    data.max_response_length=8192
    actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen3-8B}"
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000
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
    rllm.agent.max_steps=15
    rllm.agent.trajectory_timeout=600
    rllm.stepwise_advantage.enable=False
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=sc-grpo
    trainer.log_val_generations=1
    trainer.val_before_train=True
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}
    trainer.nnodes=${N_NODES:-1}
    trainer.save_freq=300
    trainer.test_freq=20
    trainer.default_hdfs_dir=null
    +rllm.agent.engine_args.max_concurrency=32
    trainer.total_epochs=${TOTAL_TRAINING_STEPS:-300}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-300}
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
    trainer.experiment_name=sc-grpo-webshop
)
COMMAND=("${PYTHON_BIN:-python3}" -m examples.webshop.train_webshop_agent "${COMMON_ARGS[@]}" "${SC_GRPO_ARGS[@]}" "$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    exec "${COMMAND[@]}"
fi
