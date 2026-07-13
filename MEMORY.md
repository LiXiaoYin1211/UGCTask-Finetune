# 工作区记忆（MEMORY）

> 本文件用于任务中断后重启时快速恢复上下文。**长期原则**记录在此，**任务进展/结果**增量记录在 `finetune/TASK.md` 的「执行日志」章节。

---

## 一、工作区

> **目录结构（2026-07-11 覆盖式更新）**

- **工作区根路径**：`/apdcephfs/private_hynnzhang/ServerFiles`

```text
ServerFiles/
├── MEMORY.md              工作区记忆（本文件，长期原则 + 环境事实）
├── finetune/              微调主体：代码/数据/日志/产物（详见 finetune/README.md）
├── fc_eval/               评测集/工具表（被 finetune 复用，训推同源）
├── model/Qwen/Qwen3___5-2B/   基座模型（~4.3G）
└── finetune_env312/       正式训练 venv（Python 3.12.13，~5.5G，唯一）
```


## 二、核心维护原则（必须遵守）

1. **TASK.md 增量更新**：`finetune/TASK.md` 是增量维护文档，所有方案变更、执行结果一律**追加**到文末「五、执行日志」，不覆盖历史内容。每次追加带**当前时间**。
2. **所有产出留在工作区内**：任何工作结果（代码、数据、模型产物、日志、环境记录等）**只保存在 `/apdcephfs/private_hynnzhang/ServerFiles` 路径下**，禁止写到工作区以外的其他位置。
3. **训练结束保存可视化产出**：任何训练/微调任务（SFT、DPO 等）结束后，**必须自动保存持久化曲线图片**（至少含 `loss`、`accuracy/reward` 等关键指标）到工作区合适位置（如 `finetune/logs/pipeline/`），并尽量同步保存 `CSV` 等结构化数据，便于复盘和报告。

## 二.5、训练编排耗时阈值规则（2026-07-11 用户设定，长期遵守）

> 每次开跑训练前，基于**实测 SFT 速度**预估「SFT + DPO 训练 + 评测」总耗时，据此选择编排：

- **总耗时 > 8h**：完成 SFT 后**先做 SFT 评测**，**暂不执行 DPO**（SFT adapter 保留，后续可单独承接 DPO）。用脚本 `finetune/src/train/run_sft_eval.sh`。
- **总耗时 ≤ 8h**：完成 SFT 后**自动接 DPO，并最终评测**。用脚本 `finetune/src/train/run_train_chain.sh ... 1`（末位 with_eval=1）。
- 当前实测（H20，本数据）：SFT ~4.7h（89~90s/step×189）、DPO ~5~6h、评测 ~10min → 合计 ~10~11h **> 8h** → 走"仅 SFT + 评测"分支。

## 三、关键背景（速览）

- 任务：Qwen3.5-2B 工具调用 QLoRA 微调（先 SFT 后 DPO），禁 think。
- 数据已就绪：`finetune/data/sft.jsonl`(1000) + `dpo.jsonl`(800)，与评测集 0 重复、0 校验错误。
- 训练脚本已落地：`finetune/src/{datagen,train,eval}/`，一键串联 `src/train/run_train_chain.sh`。
- 环境依赖：见 `finetune/requirements.txt`（torch 需按服务器实际 CUDA 安装，不可直接用本地 cu128 轮子）。
- 注意：脚本原在 Windows 本地开发，迁到 Linux 服务器需核对路径/import 顺序适配。

## 四、服务器环境事实（2026-07-10 核验）

- **服务器**：TencentOS 3.2 / **H20 96GB（sm_90）** / 驱动 535 / CUDA 12.2（nvcc 12.1）→ torch 用 **cu121**。
- **系统 Python 只有 3.9.16**，无法装 transformers 5.x（需 ≥3.10）。已用 `uv` 在工作区拉取 **CPython 3.12.13**：`.uv_python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`。
- **训练用 venv（正式，唯一）**：`/apdcephfs/private_hynnzhang/ServerFiles/finetune_env312`（Python 3.12）。
  - 解释器：`finetune_env312/bin/python`；跑串联脚本用 `PY=<该 python> bash src/train/run_train_chain.sh ...`（服务器无 `python` 命令）。
- **依赖策略**：按平台自动选最新兼容版（不锁 requirements pin），torch=cu121。
- **工作区内落盘**：`.uv_python/`(解释器) `.uv_cache/` `.pip_cache/`(缓存) `finetune_env312/`(venv)；安装脚本 `finetune/src/train/setup_env.sh`（带 flock 单实例锁）；安装日志 `finetune/logs/env_setup.log`；版本快照 `finetune/logs/env_freeze.txt`。
- **重启接续**：若 `env_setup.log` 未见 "SETUP DONE"，重跑 `bash finetune/src/train/setup_env.sh`（有锁防并发、会干净重建 venv）。
- **⚠️ cuBLASLt 崩溃修复（2026-07-10 必须保留）**：默认装的 `nvidia-cublas-cu12 12.1.3.1`（随 torch cu121）在 **H20/sm_90 上 `generate()` 会触发 cuBLASLt 整数除零 SIGFPE**（`dmesg` 见 `divide error in libcublasLt.so.12`，`F.linear` 处崩溃，无 Python traceback）。**修复：`pip install nvidia-cublas-cu12==12.4.5.8`**（有 torch 依赖冲突 WARNING，可忽略）。若 venv 被重建，务必重新升级此包，否则所有 HF 推理/评测会崩。`DISABLE_ADDMM_CUDA_LT=1` 无效。
- **⚠️ SFT 训练两大坑（2026-07-11 已修，务必保留）**：
  1. **max_length 必须够大**：数据 system+22 工具 schema 前缀就约 **7600 token**，SFT 完整轨迹最长 **9336**、DPO 最长 **8116**。原脚本 `max_length=4096` + keep_start 截断会把 assistant(label) 全截掉 → **loss 恒为 0（白训）**，且 trl 逐样本报错在截断前、不触发，冒烟会假性通过。现值：SFT `max_length=10240`；DPO `max_length=8704`、`max_prompt_length=7808`。
  2. **assistant_only_loss 需 `{% generation %}` 标记**：模型自带 `chat_template.jinja` 无此标记 → 掩码全 0。已在 `finetune/src/train/chat_template_train.jinja` 放打标记版（渲染与原模板逐字一致，仅训练用；不影响推理/评测），`train_sft.py` 加载覆盖 tokenizer。
  - 验证真学习的信号：`loss>0 且下降、grad_norm>0、mean_token_accuracy>0`（全 0 即上述坑复发）。

---

## 附·临时 — 后续重训提速方案候选（⚠️ 临时记录，方案确定后删除本节）

> 背景：当前实测 **~88~90s/step**、GPU 100%（**计算受限**）。
> 关键架构事实（决定优先级）：`Qwen3_5` 24 层 = **18 层线性注意力(gated-delta-net) + 6 层标准注意力**；6 层标准注意力**默认已用 `sdpa`**；18 层线性注意力的快路径需 **`flash-linear-attention`(fla) + `causal-conv1d`**，二者**当前均未装** → 走 `torch_causal_conv1d_update` 慢回退（modeling_qwen3_5.py:428 告警），**这是 88s/step 的主要来源**。
> ‼️ **所有方案落地前必须"小样本冒烟验证"**：改动后跑几十 step，确认 ① 无 import/编译/加载报错；② `loss>0 且只对 assistant 段算`（掩码正确，防 BUG#1/#2 复发）；③ 记录改动前后 `s/it` 确认确有提速。**不要中断正在跑的续训**，均放到"数据增强后重训"那轮再上。

候选（按性价比）：

1. **⭐【可能可行·最高收益】装 `flash-linear-attention` + `causal-conv1d`** —— 修 18/24 层线性注意力的慢回退（直击主耗时）。
   - 可行性：fla 依赖 triton(已装 3.1.0)；causal-conv1d 需 CUDA 编译(cu121/sm_90)，须与 torch 2.5.1 兼容。可能编译/版本受阻。
   - 冒烟：装后 `import` + 确认告警消失(fast path 生效) + loss 掩码正确 + s/it 明显下降；装不上则保持慢回退、不阻断。

2. **【可能可行·+25~30%】关 `gradient_checkpointing` + 降 batch 到 1~2** —— GC 省显存费时间，计算受限下关掉提吞吐；但 batch8×L10240 关 GC 几乎必 OOM（现 GC 开着已占 66G）。
   - 冒烟：降 batch 后关 GC 跑几十 step，实测不 OOM + loss 正常（batch 非提速杠杆，batch2≈batch8 wall-clock）。

3. **【可行·低风险·中收益】去 4bit 改 bf16 基座** —— 2B bf16 ~4.4G，显存足；消除每步 NF4 反量化开销。改 `build_model_tok`(去 quantization_config/prepare_model_for_kbit_training)。优化器保留 `paged_adamw_8bit` 或改 `adamw_torch`。
   - 冒烟：显存与 loss 正常、s/it 有降。

4. **【低价值·可选】FlashAttention-2** —— 仅覆盖 6 层标准注意力，且 `sdpa` 已默认，增益边际；需 `pip install flash-attn` 且 `attn_implementation="flash_attention_2"`。优先级低。

5. **【不建议】packing / group_by_length** —— 本数据样本 ~8000~9336 token 已接近 `max_length=10240`，一个窗口塞不下两条 → packing 无法打包、收益≈0，还引入 FA2 变长掩码风险；group_by_length 因共享 7600 前缀收益微。**不做**。
