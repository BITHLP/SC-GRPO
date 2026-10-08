# SC-GRPO

论文 **Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards** 的主实验代码（[论文链接](https://arxiv.org/abs/2606.18810)）。

SC-GRPO 使用模型自身生成的轨迹构建自条件教师，通过 token 级 KL 散度为 GRPO 更新分配权重。本仓库包含完整的 rLLM 训练框架、修改后的 veRL 后端，以及 Math、LiveCodeBench、AppWorld 和 WebShop 的数据准备、训练与评测入口。

## 安装

使用配备 NVIDIA GPU 的 Linux 系统，以及 Python 3.10 或 3.11。启动脚本默认使用单节点、8 张 GPU。CUDA 依赖采用 PyTorch 2.6 和 vLLM 0.8.5.post1；安装 FlashAttention 需要兼容的 CUDA 编译工具链。

```bash
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

训练使用仓库内 `verl/` 中的修改版 veRL。任务环境的安装方法见[环境配置说明](docs/environments.md)。

## 主实验

基础模型为 `Qwen/Qwen3-8B`，关闭 thinking。所有任务均为每个问题采样 8 条轨迹，每个训练批次包含 8 个问题，学习率为 `1e-6`。

| 任务 | 训练数据 | 训练样本数 | 最大响应 token 数 | 训练步数 |
| --- | --- | ---: | ---: | ---: |
| Math | DAPO-Math-17k | 4,000 | 8,192 | 500 |
| LiveCodeBench v6 | LCB v6 | 131 | 8,192 | 320 |
| AppWorld | 官方训练集 | 90 | 12,288 | 110 |
| WebShop | 从训练任务池中随机抽样 | 2,400 | 8,192 | 300 |

Math 和 WebShop 的抽样种子可配置，默认值为 42。LiveCodeBench 数据准备沿用 [SDPO](https://arxiv.org/abs/2601.20802) 的设置。

### Math：AIME 2024 和 2025

```bash
python -m examples.dapo_math.prepare_math_data --train_size 4000 --seed 42
bash examples/dapo_math/train_dapo_math.sh
```

### LiveCodeBench v6

```bash
python -m examples.livecodebench.prepare_livecodebench_data
bash examples/livecodebench/train_livecodebench.sh
```

### AppWorld

按[环境配置说明](docs/environments.md)，在独立环境中导出任务 ID 并启动 AppWorld 服务。随后在训练环境中准备数据注册表并开始训练：

```bash
export APPWORLD_TASK_IDS_FILE=/path/to/appworld-task-ids.json
python -m examples.appworld.prepare_appworld_data \
  --dataset_name appworld_official_test_normal --val_split test_normal
export APPWORLD_REMOTE_ENVIRONMENT_URLS=http://127.0.0.1:8100,http://127.0.0.1:8101
bash examples/appworld/train_appworld_agent.sh
```

### WebShop

按[环境配置说明](docs/environments.md)安装 WebShop 后端。

```bash
export WEBSHOP_ROOT=/path/to/webshop
python -m examples.webshop.prepare_webshop_data \
  --train_size 2400 --seed 42 --split_mode official \
  --dataset_name webshop_text_official
bash examples/webshop/train_webshop_agent.sh
```

## 配置

SC-GRPO 默认使用 top-64 概率和聚合后的尾部概率计算 KL 信号，token 权重为 `KL/(KL+c)`，其中 `c=max(p75,1e-4)`，探索系数为 `0.1`。组路由与教师提示词构建由训练框架实现。公开配置项使用 `rllm.sc_grpo` 前缀。

各启动脚本均支持通过 Hydra 参数覆盖配置。模型路径、GPU 数量和总训练步数也可以通过环境变量设置：

```bash
MODEL_PATH=/path/to/Qwen3-8B N_GPUS_PER_NODE=8 \
  bash examples/livecodebench/train_livecodebench.sh trainer.test_freq=16

DRY_RUN=1 bash examples/livecodebench/train_livecodebench.sh
```

`DRY_RUN=1` 打印启动命令，不启动 Ray 或训练。日志默认输出到终端。使用 SwanLab 时，自行设置 `SWANLAB_API_KEY`，并传入 `trainer.logger='[console,swanlab]'`。

准备后的数据默认存储于 `~/.cache/sc-grpo`。可通过 `RLLM_DATA_DIR` 指定共享数据目录。

## 评测

使用 Hugging Face 模型或导出的 checkpoint 进行评测，每个问题采样 8 条轨迹：

```bash
bash scripts/evaluate.sh math /path/to/exported-model
bash scripts/evaluate.sh livecodebench /path/to/exported-model
bash scripts/evaluate.sh appworld /path/to/exported-model
bash scripts/evaluate.sh webshop /path/to/exported-model
```

AppWorld 和 WebShop 使用与训练相同的环境变量。评测脚本设置 `trainer.val_only=True`，只进行评测，不更新模型参数。验证日志按数据来源记录 `avg@8` 和 `pass@8`。

导出 FSDP actor checkpoint：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir /path/to/global_step_N/actor --target_dir /path/to/exported-model
```

## 代码结构

- `rllm/engine/sc_grpo.py`：轨迹提取、分组和教师输入 token 化。
- `rllm/engine/rewrite_prompt.py`：各任务的自条件提示词。
- `rllm/trainer/verl/agent_ppo_trainer.py`：轨迹路由和教师批次构建。
- `verl/verl/workers/actor/dp_actor.py`：教师与学生概率计算、KL 加权和 actor 更新。
- `examples/`：四个主实验的入口。

## 测试

```bash
python -m pytest -q tests/test_sc_grpo_equivalence.py
python scripts/verify_release.py
```

## 致谢与许可证

本项目基于 [rLLM](https://github.com/rllm-org/rllm) 和 [veRL](https://github.com/volcengine/verl) 开发，保留了原有版权声明。许可证与上游署名见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 `verl/Notice.txt`。

## 引用

```bibtex
@misc{shan2026learningsolutionsselfconditionedcredit,
      title={Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards}, 
      author={Yingyu Shan and Yuhang Guo and Zihao Cheng and Zeming Liu and Xiangrong Zhu and Xinyi Wang and Jiashu Yao and Wei Lin and Hongru Wang and Heyan Huang},
      year={2026},
      eprint={2606.18810},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.18810}, 
}
```
