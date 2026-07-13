# FineTune — Qwen3.5-2B 工具调用 QLoRA 微调

小模型 function-calling 能力增强：SFT 提升工具选择/依赖链，DPO 对齐 `ask_user` 行为。
完整方案、规格与执行日志见 [`TASK.md`](./TASK.md)。

## 目录结构

```
finetune/
├── src/               源码（按职责三层）
│   ├── datagen/       数据生成：toolset / scene_db / prototypes / gen_sft / gen_dpo / validate
│   ├── train/         训练：data_utils / train_sft / train_dpo / chat_template_train.jinja / run_*.sh / monitor_train / tail_train
│   └── eval/          评测：eval_hf / compare_three_stage / compare_epochs / plot_curves
├── data/              训练数据（sft.jsonl / dpo.jsonl / stats.md）
├── outputs/           LoRA 产物（训练后生成：sft_lora / dpo_lora）
└── logs/              运行时日志（训练后生成）
```

## 环境

> 2026-07-11 更新：项目已从 Windows 本地开发迁移到 Linux 服务器，以下为**服务器实际运行环境**。

- **服务器**：TencentOS 3.2 / NVIDIA **H20 96GB（sm_90）** / 驱动 535 / CUDA 12.2（nvcc 12.1）→ torch 用 **cu121**。
- **虚拟环境**：`/apdcephfs/private_hynnzhang/ServerFiles/finetune_env312`（**Python 3.12.13**，用 `uv` 在工作区内拉取）。
  - 服务器无全局 `python` 命令，脚本一律用绝对路径解释器：`finetune_env312/bin/python`。
  - 一键搭建：`bash src/train/setup_env.sh`（uv + venv；解释器/缓存/venv 全落在工作区内，带 flock 单实例锁；日志见 `logs/env_setup.log`）。
- **关键依赖**：PyTorch **2.5.1+cu121**、transformers **5.13**、trl **1.8**、peft **0.19**、accelerate **1.14**、bitsandbytes **0.49**、datasets **5.0**。
- **基座模型**：`../model/Qwen/Qwen3___5-2B`；评测集/工具表：`../fc_eval/`。


## 训练配置与策略

### 通用
- 基座 Qwen3.5-2B；**QLoRA** = 4bit NF4 量化（double quant）+ bf16 计算 + LoRA。
- 优化器 `paged_adamw_8bit`；`gradient_checkpointing` 开（`use_reentrant=False`）；`bf16`；`logging_steps=1`。
- **训推一致**：每条训练样本携带全 22 工具完整 schema，`SYSTEM_PROMPT` 与 `fc_eval` 同源。
- **禁 think**：数据不含 `reasoning_content`。

### SFT（`train_sft.py`）
| 项 | 值 |
| --- | --- |
| LoRA | r=16, alpha=32, dropout=0.05, target=all-linear, bias=none |
| batch | per_device=8 × grad_accum=2 = **等效 16** |
| learning_rate | 2e-4 |
| epochs | 由脚本传入（编排默认 3，可 `--epochs` 调） |
| max_length | **10240**（覆盖 system+22 工具 schema≈7600 + 完整轨迹最长 9336；过小 + keep_start 截断会把 assistant(label) 全截掉 → loss 恒 0） |
| assistant_only_loss | **True**，依赖 `chat_template_train.jinja` 的 `{% generation %}` 标记（只对 assistant 轮算 loss） |
| packing | False |
| 保存/评估 | 默认模式：10% 验证集 + `eval_strategy=epoch` + `load_best_model_at_end(eval_loss)` 早停；**`--no-eval` 模式（编排采用）**：用满 1000 条、`save_strategy=epoch`、`save_total_limit=2`（保留最后两轮 ep2/ep3） |
| lr_scheduler / warmup / weight_decay | 未显式设置，用 trl 默认 |

### DPO（`train_dpo.py`）
| 项 | 值 |
| --- | --- |
| 起点 | 4bit 基座 + SFT LoRA adapter；`ref_model=None`（trl 用禁用 adapter 的同一模型作参考，省一份显存） |
| batch | per_device=1 × grad_accum=16 = **等效 16** |
| learning_rate | 5e-5 |
| epochs | 默认 2（`--epochs` 可调） |
| beta | 默认 0.1（串联脚本可传 0.3） |
| max_length / max_prompt_length | **8704 / 7808**（覆盖全 prompt+完成，防截断破坏训推一致） |
| 保存 | `save_strategy=epoch` |

### 编排策略（耗时阈值，见 MEMORY.md）
开跑前按实测 SFT 速度预估「SFT + DPO + 评测」总耗时：
- **> 8h**：先 SFT + SFT 评测，DPO 暂缓 → `src/train/run_sft_eval.sh`。
- **≤ 8h**：SFT 自动接 DPO 并最终评测 → `src/train/run_train_chain.sh ... 1`。
- 当前实测（H20 本数据）：SFT ~4.7h（~89s/step×189）、DPO ~5–6h、评测 ~10min → ~10–11h（**> 8h**）→ 走「仅 SFT + 评测」分支。

### 评估方式
- **统一 HF 推理路径**（`eval_hf.py`）：base / SFT / DPO 三者走同一条 transformers+peft 路径，think-off、贪心解码（temperature=0），跑 `fc_eval` 首步评测。
  - 复用 `fc_eval` 的 `SYSTEM_PROMPT` / `tools_ugc.json` / `dataset.jsonl`，口径一致。
  - 解析 Qwen3.5 特殊文本工具调用格式，输出与 Ollama runner 同构的 `raw_think_off.jsonl`，`scorer.py` 零改动复用。
- **选优 / 过拟合判断**：`compare_epochs.py`（base/ep2/ep3）、`compare_three_stage.py`（base/SFT/DPO，出对比表 + 柱状图）。
- **曲线持久化**：`plot_curves.py`（读 `trainer_state.json` → PNG + CSV）；`monitor_train.py` 实时终端面板。

## 快速上手

所有脚本从 `finetune/` 根目录运行，路径与 cwd 无关：

```bash
# 1) 生成训练数据
python src/datagen/gen_sft.py --n 1000
python src/datagen/gen_dpo.py --n 800
python src/datagen/validate.py            # 校验 schema/配对/ID/去重/配比

# 2) 训练（先 SFT 后 DPO；--smoke 走 2 步冒烟）
python src/train/train_sft.py [--smoke] [--tag _v2] [--epochs 2]
python src/train/train_dpo.py [--smoke] [--tag _v2] [--epochs 1] [--beta 0.3]

# 或用自包含串联脚本一键跑 SFT->DPO（服务器 Linux，产物校验串联，可选接评测）
PY=/apdcephfs/private_hynnzhang/ServerFiles/finetune_env312/bin/python bash src/train/run_train_chain.sh _v4 2 1 0.3 1

# 3) 评测（HF 统一推理路径，think-off）
python src/eval/eval_hf.py --slug qwen3.5-2b-hf-base
python src/eval/eval_hf.py --slug qwen3.5-2b-hf-sft --adapter outputs/sft_lora
python src/eval/eval_hf.py --slug qwen3.5-2b-hf-dpo --adapter outputs/dpo_lora
python src/eval/compare_three_stage.py    # base/SFT/DPO 三段对比报告
```

## 路径约定

每个脚本用 `ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))` 定位 `finetune/` 根，
`data/`、`outputs/`、`logs/`、`../model`、`../fc_eval` 均基于 `ROOT` 解析。
同功能组内本地 import（如 `import scene_db`）依赖「脚本所在目录即 `sys.path[0]`」，直接 `python src/<组>/<脚本>.py` 运行即可。
