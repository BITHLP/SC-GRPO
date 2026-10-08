#!/usr/bin/env bash
# Adapted from the rLLM installation script used for the experiments.
# Upstream: https://github.com/volcengine/verl/blob/v0.5.0/scripts/install_vllm_sglang_mcore.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python3}
PIP=("$PYTHON_BIN" -m pip)

# The original script installed SGLang twice; one installation is sufficient.
echo "1. Install inference frameworks and PyTorch"
"${PIP[@]}" install "sglang[all]==0.4.6.post5" --no-cache-dir
"${PIP[@]}" install torch-memory-saver --no-cache-dir
"${PIP[@]}" install --no-cache-dir \
    "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" \
    "torchaudio==2.6.0" "tensordict==0.6.2" torchdata

echo "2. Install basic packages"
"${PIP[@]}" install 'transformers[hf_xet]>=4.51.0' accelerate datasets peft hf-transfer \
    'numpy<2.0.0' 'pyarrow>=19.0.1' pandas \
    'ray[default]' codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 \
    liger-kernel mathruler blobfile xgrammar pytest py-spy pyext pre-commit ruff
"${PIP[@]}" install 'nvidia-ml-py>=12.560.30' 'fastapi[standard]>=0.115.0' \
    'optree>=0.13.0' 'pydantic>=2.9' 'grpcio>=1.62.1'

echo "3. Install FlashAttention and FlashInfer"
# Replace the original machine-specific wheel paths with local overrides or downloads.
WHEEL_DIR=$(mktemp -d)
trap 'rm -rf "$WHEEL_DIR"' EXIT
PYTHON_TAG=$("$PYTHON_BIN" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
if [[ -n "${FLASH_ATTN_WHEEL:-}" ]]; then
    "${PIP[@]}" install --no-cache-dir "$FLASH_ATTN_WHEEL"
else
    FLASH_ATTN_NAME="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl"
    curl --fail --location --retry 3 \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_NAME}" \
        --output "$WHEEL_DIR/$FLASH_ATTN_NAME"
    "${PIP[@]}" install --no-cache-dir "$WHEEL_DIR/$FLASH_ATTN_NAME"
fi
if [[ -n "${FLASHINFER_WHEEL:-}" ]]; then
    "${PIP[@]}" install --no-cache-dir "$FLASHINFER_WHEEL"
else
    FLASHINFER_NAME="flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
    curl --fail --location --retry 3 \
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/${FLASHINFER_NAME}" \
        --output "$WHEEL_DIR/$FLASHINFER_NAME"
    "${PIP[@]}" install --no-cache-dir "$WHEEL_DIR/$FLASHINFER_NAME"
fi

echo "4. Install and fix OpenCV"
"${PIP[@]}" install opencv-python opencv-fixer
"$PYTHON_BIN" -c 'from opencv_fixer import AutoFix; AutoFix()'

echo "5. Install the modified veRL backend"
"${PIP[@]}" install --no-deps -e "$REPO_ROOT/verl"

echo "6. Install rLLM"
"${PIP[@]}" install -e "$REPO_ROOT"
"${PIP[@]}" install codetiming
