# 微调任务说明（FineTune Task）

> 本文档**增量维护**，持续记录微调任务的方案、清单与执行结果。最新进展追加到文末。

---

## 一、微调方案

| 项          | 内容                                                               |
| ---------- | ---------------------------------------------------------------- |
| **微调模型**  | `qwen3.5:2b`                                                      |
| **教师模型**  | Claude-Opus-4.8（生成 SFT/DPO 训练数据）                                  |
| **基座工具表** | 复用 `../fc_eval/tools_ugc.json`（真实 UGC Runtime 22 工具）              |
| **场景素材**  | 复用 `../fc_eval/item_table.json` + `../MyDataFiles/06_世界场景状态.md`   |
| **数据格式**  | OpenAI `tool_calls` wire format（对齐 `../MyDataFiles/04_轨迹格式约定.md`） |

### 微调背景（来自评测与消融实证）

1. **工具选择准确率偏低**：2b 工具选择准确率仅 55%（think 开），9b 提升到 77.5% → 主要是**模型容量/语义理解**问题。
2. **信息不足时反问偏好不足**：信息不足/指代不明时，2b 几乎不触发 ask_user（C 类召回 0%）。提示词消融证明——
   - 强化 prompt 后 2b 仅 0/8→3/8，9b 0/8→6/8（think 开）；
   - 即**不是工具描述不佳，而是模型行为决策/指令遵循能力不足**；
   - **提示词工程能缓解但有天花板**（2b 强化后仍 5/8 不触发）。

### 微调目的

增强小模型的 function-calling 能力，并对齐当前任务偏好：

- **目标 A**：提升 **工具选择** 准确率（选对工具、遵守前置依赖）；
- **目标 B**：强化 **「信息不足 → 优先 ask_user」** 的行为偏好。

### 方法路线

- **SFT**：用教师模型生成的**正确多步轨迹**作监督，重点覆盖 **B 类前置查询**（先查询拿 unique_id 再操作）与 **A 类工具选择**。
- **DPO**：用 **chosen/rejected 偏好对**对齐行为，重点是 **C 类 ask_user**（chosen=调 ask_user，rejected=瞎猜执行/纯文本提问）+ 工具选择的困难负例（错选/跳依赖/ID 混用）。

### 评测闭环

训练后用 `../fc_eval/` 同一套 40 例评测集 + scorer 重测 2b，对比微调前后的：工具选择、必填完整、ask_user 召回、首步整体正确率。**训练数据与评测集严格隔离，防止数据泄漏。**

---

## 二、所需资源 / 重大问题

**核心待解决问题：训练数据构建。** 以 Claude-Opus-4.8 为教师模型，基于真实工具表 + 场景，生成：

- SFT 正例（正确轨迹）
- DPO 偏好对（chosen vs rejected）

数据产物统一放置于 `finetune/data/`。

---

## 三、数据构建原则与规格（已确定，2026-06-30）

### 核心原则

1. **SFT 主攻领域内高频工具选择模式的记忆与复用**：把"哪种情况选哪个工具、必须先查再改"的轨迹喂够，提升工具选择准确率。本质是强化领域高频模式，**不增强语义理解**。**多步依赖链作为显式训练目标**（loss 覆盖整条轨迹的所有 assistant 轮，而非单步 top-1 命中）。
2. **拉高 ask_user 召回的同时防矫枉过正**：不能训成"逢歧义就问"，不过度牺牲 Agent 自主性。数据质量高于数量；**B 类「该问:不该问 = 1:1」是硬约束**。
3. **think/reasoning_content 不进数据，训推链路必须一致**：游戏运行时实时性强，当前版本做**禁 think** 微调，**基线用之前 think-off 的评测结果**。

### 三项执行决策

- **生成方法**：混合式——教师模型（Claude-Opus-4.8）写高质量种子原型，Python 从场景采样、合成一致 tool 结果、自动校验。保证 schema 与 ID 一致性。
- **推进节奏**：先 pilot（~50 SFT + ~40 DPO）→ 用户确认质量 → 扩全量。
- **落盘格式**：框架中立的原生 OpenAI messages JSONL；DPO 为 `{prompt, chosen, rejected}`。框架转换脚本留到选定微调框架时再加。

### 四个修正点

1. **数据泄漏去重**：训练 query 与 `../fc_eval/dataset.jsonl` 40 例严格不重叠（归一化文本 + 语义近似双重查重）。
2. **低频工具放宽**：核心写/改/查工具 ≥20 条；`task`/`load_skill`/`unreal_info`/`list_skills`/`ugc_analyze_scene_capture` 放宽至 ≥5~8 条。
3. **"复杂"类定义**：SubAgent(task) 分治 / ≥5 步依赖链 / 跨轮指代消解，三者之一。
4. **DPO chosen 为 off-policy**（教师直接写）：v1 已知简化，后续可升级 on-policy（从 SFT 后模型采样再判优）。

### SFT 规格（1000 条）

- 格式：`{"messages":[system,user,assistant(tool_calls),tool,assistant...], "tools":[...]}`，严格遵守 `../MyDataFiles/04`（tool_calls、arguments 为字符串化 JSON、tool_call_id 配对、role 交替）。
- 配比：**A 工具选择/依赖链 65% · B ask_user 25% · 通用对话防遗忘 10%**。
- A 内部难度：简单 35% / 多步 50% / 复杂 15%（多步依赖链为核心）。
- B 内部：**该问 : 不该问 = 1 : 1**。
- 工具覆盖：核心 ≥20 条，低频 ≥5~8 条。
- 4 类标杆原型：A1 query→detail→catalog→place 完整链；A2 检索+语义过滤+删除；B1 模糊→ask_user；B2 "附近"→player_context 自查不问。

### DPO 规格（800 对）

- 格式：`{"prompt":[...共享前缀], "chosen":[...], "rejected":[...]}`，chosen/rejected 仅在关键决策点分叉，rejected 的 tool 结果用 `../MyDataFiles/05` 真实错误串。
- 覆盖矩阵：该问不问→ask_user 22% · ID幻觉(item_table_id↔unique_id) 20% · 错选工具(place↔create) 18% · 跳过前置依赖 15% · 过度触发ask_user(反向) 13% · 格式/必填 7% · 越界/幻觉工具名 5%。
- 约束：ask_user #1(该问) 与 #5(过度追问) 成对；困难负例含形似错工具 / 依赖链差一步 / 参数级细错(id 填成上一步 unique_id) / 跨轮指代漂移。

---

## 四、训练环境与配置（H20 服务器，已核验 2026-07-10）

### 硬件 / 环境（实测通过）

| 项            | 值                                                                                              | 核验                 |
| ------------ | ---------------------------------------------------------------------------------------------- | ------------------ |
| 服务器 / OS     | TencentOS Server 3.2（RHEL 系）                                                                   | ✅                  |
| GPU          | **NVIDIA H20，Hopper sm_90，96GB（97871 MiB）**                                                    | ✅                  |
| 驱动 / 系统 CUDA | 535.247.01 / CUDA 12.2（nvcc 12.1）→ torch 用 **cu121**                                           | ✅                  |
| 虚拟环境         | `/apdcephfs/private_hynnzhang/ServerFiles/finetune_env312`（Python **3.12.13**，uv 拉取，工作区内）      | ✅                  |
| PyTorch      | **2.5.1+cu121**（原生支持 H20 sm_90）                                                                | ✅ bf16 matmul 实测通过 |
| 关键依赖         | transformers 5.13.0 · trl 1.8.0 · peft 0.19.1 · accelerate 1.14.0 · bitsandbytes 0.49.2 · datasets 5.0.0 | ✅                  |
| ⚠️ cuBLAS 修复 | 必须 `nvidia-cublas-cu12==12.4.5.8`（默认 12.1.3.1 在 H20 上 `generate()` 触发 SIGFPE 崩溃，见五、2026-07-10 19:01） | ✅                  |

> **环境沿革**：本任务最初在本地 Windows + RTX5060（Blackwell sm_120，8GB，torch 2.11.0+cu128）开发；2026-07-10 迁移到 H20 服务器（sm_90），改用 cu121 构建的 torch，并用 uv 在工作区内拉取 CPython 3.12.13 新建 venv。详细迁移与核验过程见「五、执行日志 2026-07-10」及 `MEMORY.md` 四。

### 模型（已下载并核验）

- 路径：`../model/Qwen/Qwen3___5-2B/`（ModelScope 命名，含三下划线）。
- 架构：`Qwen3_5ForConditionalGeneration`，**多模态**（嵌套 config）：`text_config`(hidden 2048 / 24 层 / 8 heads / vocab 248320) + `vision_config`(hidden 1024)。
- 核验：config / tokenizer / chat_template 均成功加载（AutoConfig+AutoTokenizer，trust_remote_code=True，0.5s）。
- **注意**：本任务仅做纯文本工具调用微调，应只训练语言塔，**冻结 vision 塔**（LoRA target 限定 text_config 下的线性层，不挂 vision/merger）。

### 框架选型：**transformers + peft + trl 直写脚本**（已定，2026-06-30 改）

> **决策变更**：原计划 LLaMA-Factory，经评估后改为直写脚本。下为理由。

- **为何不用 LLaMA-Factory**：`qwen3_5` 是很新的多模态架构 + transformers 5.12.1 属前沿组合，框架稳定版很可能未注册该 `model_type`/对话模板。一旦不识别，会卡在框架内部报错——框架抽象层厚、最难定位，恰好抵消其"提效"价值。
- **直写 ≠ 造轮子**：仍站在 `trl` 的 `SFTTrainer`/`DPOTrainer` 上（已封装 loss mask、参考模型、偏好损失等最易写错的部分），只把"YAML 配置"换成"~80 行透明 Python"。
- **直写的收益**：① 报错落在自己代码里，可控；② 多模态模型"只训语言塔"的精细控制（LoRA target 正则）框架反而难做；③ 任务轻（2B + 1800 条 + 单卡 QLoRA），训练循环极标准；④ **简历价值**——面试能讲清 QLoRA/DPO loss/LoRA 挂载层每一步。
- 备选：若直写遇阻，回退 ms-swift。

### 直写技术栈（已核验）

- `trl` 升级到 **1.7.0**（旧版 0.17 引用了 transformers 5.x 已移除的 `MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`，与 transformers 5.12 不兼容；1.7.0 兼容）。
- **模型加载用 `AutoModelForCausalLM`**：实测会自动剥离 vision 塔，加载为 `Qwen3_5ForCausalLM`（仅语言模型，无 vision 模块）——故"冻结 vision"问题天然不存在，无需手动处理。
- 模型为**混合注意力架构**：18 层线性注意力(`in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj`) + 6 层标准注意力(`q/k/v/o_proj`) + MLP(`gate/up/down_proj`)。LoRA target 用 `all-linear`（peft 自动挂所有 4bit 线性层、排除 lm_head）。
- **chat_template 适配**：Qwen3.5 模板要求 `tool_calls[].function.arguments` 为 **dict**，而数据按 OpenAI 标准存为字符串化 JSON → `data_utils.fix_tool_args` 在加载时 parse 回 dict。
- **⚠️ import 顺序坑（关键）**：Windows + torch(cu128) 下，必须**先 import `datasets`/`trl`，再 import `torch`**。否则 torch 先加载会致随后 pyarrow 原生扩展 access violation 段错误（exit 139，无 Python 异常）。两个训练脚本已按此顺序写并注释。

### 训练脚本（已落地于 `finetune/`）

| 文件               | 作用                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `data_utils.py`  | 加载 SFT/DPO JSONL → trl 可吃的 Dataset，含 `fix_tool_args`                                                               |
| `probe_model.py` | 模型结构探查（4bit 加载 + 打印线性层名）                                                                                           |
| `train_sft.py`   | QLoRA SFT，`assistant_only_loss=True` 只算 assistant 轮 loss                                                           |
| `train_dpo.py`   | 在 SFT 基础上 DPO，`ref_model=None`（peft 禁 adapter 作参考省显存）                                                              |
| 用法               | `python src/train/train_sft.py [--smoke]` / `python src/train/train_dpo.py [--smoke]`（`--smoke` 走 2 步 + 少量样本验证全链路） |

### QLoRA 配置（8GB / 2B / 禁 think）

| 参数          | 建议值                                            | 理由                           |
| ----------- | ---------------------------------------------- | ---------------------------- |
| 量化          | 4bit **NF4** + double_quant                    | 2B 4bit 权重 ~1.5GB，给激活/优化器留空间 |
| 计算精度        | **bf16**                                       | Blackwell 原生支持，比 fp16 稳      |
| LoRA        | r=**8~16**，alpha=16~32，dropout 0.05            | 小数据防过拟合，r=8 起步               |
| target      | **all-linear**（CausalLM 加载已自动剥离 vision，无需手动冻结） | 工具调用需结构化输出，语言塔全挂             |
| max_seq_len | **1024**                                       | 多步轨迹够用，省显存                   |
| batch / 累积  | batch **1** + grad_accum **8~16**              | 等效 batch 8~16                |
| 梯度检查点       | **开**                                          | 8G 必开，时间换显存                  |
| 优化器         | **paged_adamw_8bit**                           | 防显存峰值 OOM                    |
| 预估显存        | 5~~7GB（留 1~~2G 余量）                             |                              |

**训练顺序**：先用 `data/sft.jsonl` 训 LoRA → 在 SFT 产物上用 `data/dpo.jsonl` 做 DPO（DPO 需 policy+reference 两份模型，8G 更紧，届时可降 max_len→768 或 r→8）。  
**评测闭环**：训练后用 `../fc_eval/` 同套 40 例（禁 think 那组为基线）重测对比。

---

## 五、执行日志（增量追加）

### 2026-07-10 17:43 — 迁移到 CephFS 服务器：环境核验 + 脚本路径适配检查

> 工作区已迁到 `/apdcephfs/private_hynnzhang/ServerFiles`（下文所有路径以此为根）。工作区记忆见 `MEMORY.md`。

**1) 服务器硬件/系统核验（实测）**

| 项 | 值 | 与本地(RTX5060)差异 |
| --- | --- | --- |
| OS | TencentOS Server 3.2（RHEL 系） | Windows → Linux |
| GPU | **NVIDIA H20，Hopper sm_90，97871 MiB(~96GB)** | 8GB → 96GB，架构 sm_120→sm_90 |
| 驱动 / 运行时 CUDA | 535.247.01 / CUDA 12.2；nvcc 12.1 | cu128 → **cu121** |
| 系统 Python | **3.9.16**（唯一系统版本，无 3.10~3.12，无 conda/module python） | 本地 3.10.5 |
| 系统预装 torch | `torch 2.1.2+cu121`，`cuda.is_available()=True`，**bf16 matmul 实测通过** | — |

**2) 关键阻塞与决策**

- ❗ **系统 Python 3.9 无法满足 `requirements.txt`**：实测 `transformers 5.x`（及 `numpy 2.2.6`）强制 `Requires-Python >=3.10`，py3.9 封顶只能装 transformers 4.57.6。安装器仅能提供 Python 3.14.3（过新，torch/bitsandbytes 无 cp314 wheel，不可用）。
- ✅ **解决**：用 `uv` 在**工作区内**拉取独立 **CPython 3.12.13**（`./.uv_python/`），据此新建 venv **`finetune_env312`**。
- ✅ **依赖策略（经确认）**：按平台自动选**最新兼容版**，不强锁 `requirements.txt` 的 pin；torch 换 **cu121**（适配 H20/CUDA12.1）。所有缓存/解释器/venv 全部落在工作区内（`.uv_python/`、`.uv_cache/`、`.pip_cache/`、`finetune_env312/`）。
- 依赖安装脚本落盘：`finetune/src/train/setup_env.sh`；后台执行日志：`finetune/logs/env_setup.log`；版本快照将写入 `finetune/logs/env_freeze.txt`。

**3) 脚本路径适配检查（Windows→Linux）结论**

- ✅ 所有脚本用 `__file__` 自定位 `ROOT` + `os.path.join`，**无硬编码 Windows 路径**（`C:/Users/...` 仅出现在文档，代码中无）；`MODEL=ROOT/../model/Qwen/Qwen3___5-2B` 路径存在。
- ✅ `run_train_chain.sh` 为纯 POSIX、脚本自定位、通过 `PY` 环境变量指定解释器 —— 跨平台安全。
- ⚠️ **注意点（非阻塞）**：
  1. 服务器**无 `python` 命令**（只有 `python3`）。跑串联脚本必须显式指定：`PY=/apdcephfs/private_hynnzhang/ServerFiles/finetune_env312/bin/python bash src/train/run_train_chain.sh ...`。
  2. 训练脚本"先 import datasets/trl 再 import torch"的注释是 Windows+cu128 规避 pyarrow 段错误用；Linux 下无害，保留即可。
  3. 训练脚本 docstring/超参注释仍写"8GB/RTX5060"，实际 **H20 96GB 显存充裕**：`max_length` 已是 4096，后续可上调 batch/关梯度检查点/提高 LoRA r 以提速（本次不改，先跑通）。`--smoke` 冒烟入口可用。

**4) 依赖安装：完成（18:01 SETUP DONE）**

venv `finetune_env312`（Python 3.12.13，~5.6GB），依赖按平台自动选最新兼容版装成，实测 import 全通过：

| 库 | 版本 | 库 | 版本 |
| --- | --- | --- | --- |
| torch | **2.5.1+cu121**（原生支持 H20 sm_90） | datasets | 5.0.0 |
| transformers | 5.13.0 | accelerate | 1.14.0 |
| peft | 0.19.1 | safetensors | 0.8.0 |
| trl | 1.8.0 | tokenizers | 0.22.2 |
| bitsandbytes | 0.49.2（import OK，CUDA 检测通过） | numpy | 2.5.1 |

- **GPU 校验（新 venv 实测）**：`torch.cuda.is_available()=True`，device=H20 sm_90，**bf16 matmul OK**。
- 版本快照：`finetune/logs/env_freeze.txt`（83 行，pip freeze 全量）。
- 与本地 pin 的差异（因选最新兼容版）：torch 2.11→**2.5.1+cu121**、transformers 5.12.1→**5.13.0**、trl 1.7.0→**1.8.0**、numpy 2.2.6→**2.5.1**；peft/bnb/datasets/accelerate/safetensors/tokenizers 与 pin 一致。

**5) 端到端路径/加载验证（新 venv 实测，Windows→Linux 适配确认）**

- `data_utils.py` 直接运行 OK：SFT 样本含完整多步 `system/user/assistant(tool_calls)/tool/...` 轨迹；DPO 样本 `prompt/chosen/rejected/tools` 结构正确 → 数据路径与 `fix_tool_args` 在 Linux 正常。
- 模型加载 OK（transformers 5.13 + trust_remote_code）：config=`Qwen3_5Config`，tokenizer=`Qwen2Tokenizer`（vocab 248044）。
- 结论：脚本路径/加载链路在服务器 Linux 下**无需改动即可用**（跑串联脚本记得传 `PY=.../finetune_env312/bin/python`）。

**本轮小结（2026-07-10 18:01）**：服务器环境核验 + 依赖安装 + 路径适配检查**全部完成，环境就绪可训练**。下一步（待启动）：先跑 `--smoke` 冒烟（`PY=<venv> bash src/train/run_train_chain.sh _smoke ...` 或单独 `train_sft.py --smoke`）验证 QLoRA 全链路，再跑全量 SFT→DPO→评测。注意 H20 96GB 显存充裕，后续可上调 batch/关梯度检查点/提高 LoRA r 提速。

### 2026-07-10 19:01 — 原生 base 模型 HF 评测（think-off，40 例）：完成，含关键崩溃修复

> 目标：跑通 `eval_hf.py` 的 HF 统一推理路径，为微调三段对比（base/SFT/DPO）建立 **HF 4bit base 基线**。

**1) 关键阻塞与修复：cuBLASLt 除零 SIGFPE（H20/sm_90）**

- 现象：`eval_hf.py --slug qwen3.5-2b-hf-base` 模型加载成功（`Qwen3_5ForCausalLM`，确认 CausalLM 自动剥离 vision 塔），但一进 `generate()` 即**静默崩溃**，0 例写出、无 Python traceback。
- 定位：`dmesg` 见 `traps: python[...] trap divide error ... in libcublasLt.so.12`；`-X faulthandler` 栈指向 `torch/nn/modules/linear.py:125 forward`（`F.linear`）← `modeling_qwen3_5.py:1667` ← 生成 prefill。即 **cu121 自带 cuBLASLt 12.1.3.1 在 Hopper(sm_90) 上对特定 GEMM 形状触发整数除零（SIGFPE / `Fatal Python error: Floating-point exception`）**。之前 MEMORY 记的"bf16 matmul 通过"是简单方阵，未触发此形状 bug。
- 排除项：`DISABLE_ADDMM_CUDA_LT=1` **无效**（崩溃的 `F.linear` bias=None，走 matmul 的 Lt 路径而非 addmm Lt，该 env 不覆盖）。
- **修复（已生效）**：`pip install nvidia-cublas-cu12==12.4.5.8`（原 12.1.3.1）。CUDA 12.x 内 ABI 兼容，torch 运行时直接 dlopen 新 `libcublasLt.so`（会有 torch 对 cublas==12.1.3.1 的 pip 依赖冲突 WARNING，可忽略，运行时不校验版本）。升级后 2 例探针 + 全量 40 例均跑通、无崩溃。

**2) base 评测结果（`fc_eval/results/qwen3.5-2b-hf-base/`，think-off，贪心，HF 4bit NF4）**

| 指标 | 总体(n=40) | A简单(16) | B多步(16) | C模糊(8) |
| --- | --- | --- | --- | --- |
| 工具选择准确率 | **50.0%** | 68.8% | 50.0% | 12.5% |
| 首步整体正确率 | **45.0%** | 68.8% | 37.5% | 12.5% |
| 参数schema合法率 | 94.4%(36) | 100% | 87.5% | 100%(4) |
| 必填完整率 | 95.0%(20) | 100%(11) | 87.5%(8) | 100%(1) |
| 参数值正确率 | 95.0%(20) | 100% | 87.5% | 100% |
| 发起工具调用率 | 90.0% | 100% | 100% | 50.0% |
| 工具幻觉率 | 0.0% | 0% | 0% | 0% |
| 越界/禁用率 | 2.5% | 0% | 6.2% | 0% |

- **ask_user（C 类）召回率 = 0.0%**（tp=0 / fp=0 / fn=8）：8 例该问的全未触发 ask_user（4 例瞎调工具、4 例 NO_TOOL_CALL），与早期 Ollama 基线"C 类召回 0%"结论一致 → 印证微调**目标 B**（强化 ask_user）的必要性。
- **工具选择 50% / 首步 45%**：A 简单尚可(68.8%)，B 多步依赖链掉到 37.5%、C 模糊仅 12.5% → 印证微调**目标 A**（多步依赖链 + 工具选择）的空间。
- 幻觉率 0%、schema/必填/值正确率均 ≥87.5%：说明**格式/参数能力本就不差，短板在"选哪个工具 + 该不该问"的决策**，与 TASK 微调动机完全吻合。

**3) 口径说明（防混比）**：本 base 为 **HF 4bit NF4 同一推理路径**跑出的新基线，与后续 SFT/DPO 完全可比；**与早期 Ollama(Q8_0) 基线口径不同，不要混比**。

**本轮小结（2026-07-10 19:01）**：修复 cuBLASLt SIGFPE 崩溃后，原生 base 模型 HF 评测跑通并落盘（`results/qwen3.5-2b-hf-base/{raw,scored}_think_off.jsonl`、`metrics.json`）。HF base 基线：**工具选择 50% / 首步正确 45% / ask_user 召回 0%**。下一步：跑 SFT→DPO 训练，再用同路径 `eval_hf.py --adapter ...` 评测 SFT/DPO，最后 `compare_three_stage.py` 出三段对比（注意：三段评测须复用已升级的 cublas 12.4.5.8 环境）。

### 2026-07-11 10:52 — 首次启动 SFT+DPO 训练：定位并修复两处致命 bug（否则 SFT 白训）

> 本轮从"环境就绪"推进到"真正开跑训练"。启动前做 SFT 冒烟，发现 **loss/grad_norm/accuracy 全为 0**，深挖出两个必须修的问题，修复后确认真实学习，已后台启动全量链路。

**1) 环境复核（无需重配）**：venv `finetune_env312` 依赖齐全 —— torch 2.5.1+cu121(cuda OK) / transformers 5.13.0 / trl 1.8.0 / peft 0.19.1；**cuBLAS 仍为修复版 12.4.5.8**（H20 SIGFPE 修复保留）；H20 96GB 空闲。数据 SFT 1000 / DPO 800 就位。

**2) BUG#1：chat_template 缺 `{% generation %}` 标记 → assistant_only_loss 掩码全 0**
- 现象：SFT 冒烟 `loss=0, grad_norm=0, mean_token_accuracy=0`。
- 定位：模型自带 `chat_template.jinja` 不含 trl `assistant_only_loss` 所需的 `{% generation %}...{% endgeneration %}` 块；直接 `apply_chat_template(return_assistant_tokens_mask=True)` 得 `assistant_masks` 全 0。
- 修复：在 **finetune 工作区内**放一份打标记的训练模板 `src/train/chat_template_train.jinja`（仅把 assistant 生成区 content+tool_calls+`<|im_end|>` 包进 `{% generation %}`，think 前缀留在 prompt 侧以对齐禁-think 推理），`train_sft.py` 加载时覆盖到 tokenizer。**校验 30/30 渲染与原模板逐字一致**（不改推理/评测输出，训推一致），掩码变为非 0。
- 备注：其实 trl 1.8 也会自动 `get_training_chat_template` 注入标记，但仍需配合 BUG#2 才有效。

**3) BUG#2（真凶）：`max_length=4096` + keep_start 截断把 assistant(label) 全截掉**
- 关键统计：SFT 样本 **system+22 工具 schema 前缀就 ~7600 token**，首个 assistant token 位置 p50=7695，样本总长 p50=8446 / max=9336。`max_length=4096` + 默认 keep_start → **100% 样本截断后 label=0（全 -100）→ loss 恒为 0**。trl 的"无 assistant token"逐样本报错发生在**截断前**，故未触发，掩盖了问题（冒烟假性 SMOKE OK）。
- 修复：`train_sft.py` `max_length` 4096→**10240**（覆盖 max 9336）；`train_dpo.py`（DPO prompt ~7703、prompt+完成 max 8116）`max_length` 4096→**8704**、`max_prompt_length` 3584→**7808**（否则 DPO 对 prompt keep_end 截断会砍掉开头工具 schema）。
- 提速：H20 96GB 充裕，SFT `batch→8 / grad_accum→2`（等效批量仍 16）。**实测：训练全程 GPU 100% 利用率、计算受限**（每步 16×~9000 token 的 FLOPs，共享 7600-token 工具 schema 前缀占大头），batch 大小非提速杠杆（batch 2 与 8 wall-clock 近似），故不再上调；grad ckpt 保留（关掉可省 ~30% 但有 OOM 风险，不值当）。

**4) 修复后验证（SFT 冒烟）**：`loss 0.3275→0.2691` 递减、grad_norm 3.72/1.74、mean_token_accuracy 0.91/0.78、entropy 非 0，`num_tokens≈8748/样本`（确认用完整序列）。**真实学习信号恢复**。

**5) 已后台启动全量链路（2026-07-11 10:59，pid 13479）**
- 命令：`HF_HOME/HF_DATASETS_CACHE` 指向工作区内（`.hf_home` / `.hf_datasets_cache`，遵守"产物只留工作区"）；`PY=<venv> nohup bash src/train/run_train_chain.sh _v1 3 2 0.1 1`（SFT 3 轮 / DPO 2 轮 / beta 0.1 / with_eval=1 训练后自动评测）。
- 产物：`outputs/sft_lora_v1`、`outputs/dpo_lora_v1`；评测 slug `qwen3.5-2b-hf-sft_v1` / `qwen3.5-2b-hf-dpo_v1`（落 `fc_eval/results/`）。日志：`logs/pipeline/chain_{status,sft,dpo,eval_*}.log`、`chain_full_v1.log`。
- 起跑实测：SFT 共 **189 步**；step1 loss 0.3336 / acc 0.91；GPU **55GB/96GB**、util 100%；SFT 单段 ETA ~4.7h（计算受限）。
- 待办（训练完成后核对）：三段对比 `compare_three_stage.py`，看 SFT/DPO 相对 base（工具选择 50%/首步 45%/ask_user 召回 0%）的提升；重点看目标 A（工具选择/多步依赖）与目标 B（ask_user 召回）。

### 2026-07-11 11:09 — 按「>8h 阈值规则」改编排：仅 SFT+评测，DPO 暂缓

> 用户设定耗时阈值规则（已写入 `MEMORY.md` 二.5）：>8h 则 SFT 后先评测、暂不做 DPO；≤8h 则自动 SFT→DPO→评测。

**耗时预估（基于实测 SFT 速度）**
- SFT：**89~90s/step × 189 步 ≈ 4.7h**（10:59 起，约 15:40 完成）。
- DPO：batch1×accum16×2轮=100 步；单样本需 policy(chosen+rejected 前+反向)+ref(chosen+rejected 前向)，计算量约 SFT 单样本 **~2.4×**，1600 样本·轮 ≈ **5~6h**。
- 评测：base 实测 40 例约 3~4min/次 → 两次 ~10min。
- **合计 ≈ 10~11h > 8h**（即便 DPO 保守取 4.7h，总计 ~9.4h 仍 >8h；DPO 单样本必重于 SFT，不可能更快）。

**决策与动作**：走「仅 SFT + SFT 评测」分支。
- 停掉原 with_eval 全量链路（pid 13479，当时仅 5/189 步，损失极小）。
- 新增脚本 `src/train/run_sft_eval.sh`（SFT→校验 adapter→SFT 评测+scorer，不含 DPO；SFT adapter 保留供日后单独承接 DPO）。
- 已后台启动（**2026-07-11 11:09，pid 15940**）：`PY=<venv> nohup bash src/train/run_sft_eval.sh _v1 3`，HF 缓存仍指向工作区内。日志 `logs/pipeline/chain_sft_only_v1.log`、`chain_{status,sft,eval_sft,score_sft}.log`。
- 产出：`outputs/sft_lora_v1` + 评测 `fc_eval/results/qwen3.5-2b-hf-sft_v1`。
- DPO 承接命令（后续需要时）：`PY=<venv> python src/train/train_dpo.py --tag _v1 --epochs 2 --beta 0.1`。

### 2026-07-11 11:28 — 采纳"不开 eval_loss 早停，改逐轮 checkpoint + fc_eval 选优"

> 结论：本任务真实目标由 fc_eval 衡量（工具选择/ask_user 等），eval_loss 与之弱相关；且开早停要牺牲 10% 精心配比小数据、粒度粗噪声大。故保持 `--no-eval` 用满 1000 条，用下游真实指标做逐轮模型选择作为过拟合防线。

**落实（均不影响正在运行的 pid 15942 训练进程）**：
1. **注释/文档同步**：`train_sft.py` 过时 docstring（trl1.7/max_length4096/batch1）与 `--no-eval` 的"8GB"说明已更新为现状（trl1.8 / max_length 10240 / batch8×accum2 / 96GB）；并澄清"默认启用早停，编排脚本靠 `--no-eval` 关闭"。
2. **新增逐轮对比脚本 `src/eval/compare_epochs.py`**：读 base/ep2/ep3 的 `metrics.json`，出 `fc_eval/results/compare_sft_epochs.md`（总体+分类指标表），按 首步正确率>工具选择>ask_user 召回 选优，并给过拟合信号（末轮劣于更早轮即提示）。
3. **新增等待型编排 `src/train/eval_sft_epochs.sh`**：`WAIT_PID` 轮询等 SFT 主链路结束→校验 ep3 评测→自动补评保留的 ep2 checkpoint（slug `qwen3.5-2b-hf-sft_v1_ep2`）→跑 compare_epochs。已后台启动（**11:28，pid 18419**，`WAIT_PID=15942`），GPU 不与主任务争用（等其结束才评）。日志 `logs/pipeline/watch_eval_ep2_v1.log`、`chain_{eval,score}_sft_ep2.log`、`chain_compare_epochs.log`。
- 依据：`--no-eval` 下仍 `save_strategy=epoch` + `save_total_limit=2` → 训练结束保留 ep2/ep3 两个 checkpoint（ep1 被淘汰），ep3=根适配器（主链路已评）。
- 产出：`fc_eval/results/{qwen3.5-2b-hf-sft_v1, ...sft_v1_ep2}/metrics.json` + `compare_sft_epochs.md`（选优 + base 对比）。

### 2026-07-11 11:31 — 训练可视化 + 确认自动评测无需人工

- **可视化（非侵入，无需重启当前训练）**：新增 `src/train/monitor_train.py`，仅解析 `logs/pipeline/*.log` 渲染：阶段/进度条/已用+ETA/s-it/最新 loss·grad_norm·acc/loss·acc 趋势火花线/GPU；`--watch N` 实时刷新、`--png` 导出 loss 曲线（matplotlib 已装；tensorboard 未装）。实测当前 run：step11/189、loss 0.019、acc 0.995、GPU 66G/97G。快照 `logs/pipeline/sft_curve_v1.png`。
- **自动评测**：已确认全自动、无需人工——`run_sft_eval.sh`（SFT 完自动接 ep3 评测+scorer）+ 等待型 `eval_sft_epochs.sh`（pid 18419，WAIT_PID 轮询训练结束后自动补评 ep2 并出 compare）。全程后台 nohup，无交互点击。
- 备注：SFT loss 很快降到 ~0.02/acc~0.99（工具调用格式高度模板化），进一步印证逐轮选优防过拟合的必要性。

### 2026-07-11 11:42 — 训练曲线持久化策略（补齐）

- 现状盘点：`report_to=[]`（无 TensorBoard/W&B/CSV 自动记录）；逐步指标持久在 `chain_{sft,dpo}.log`(文本) 与 checkpoint/`trainer_state.json`(结构化 log_history)；**但管线无"自动出曲线图"步骤**（仅 `monitor_train.py --png` 手动快照）。
- 补齐：新增 `src/eval/plot_curves.py`（SFT/DPO 通用）：优先读 `trainer_state.json` 的 log_history(权威，含 eval_loss 若有)，无则回退解析日志；产出 `logs/pipeline/{stage}_curve{tag}.png` + `.csv`（SFT:loss+token_acc；DPO:loss+rewards_acc+margins）。
- 已烘焙进 `run_train_chain.sh`（SFT/DPO 各阶段结束后自动出图，`|| true` 不阻断）。
- 当前 SFT-only run（run_sft_eval.sh 运行中不宜改）：另起非侵入 waiter（**pid 24075**）等 pid 15942 结束后自动出 `sft_curve_v1.png/csv`（届时用 trainer_state.json）。
- 待办：run_sft_eval.sh 在本 run 结束后，可补加同样的 plot 调用（当前运行中不改）。

### 2026-07-11 14:40 — 用户决策：提前结束 SFT、采用 ep2 为最终产物；SFT 评测完成，发现严重格式退化

> 训练在 step 127/189（epoch 2.0 刚过）时，loss 早自 epoch≈1.4 起饱和至 ~1e-4、acc=1。用户决策：提前结束、以 **ep2(checkpoint-126)** 为 SFT 最终产物并做评测。

**1) 停训与产物固化**
- 强制停止训练与全部编排/waiter 进程（run_sft_eval.sh 15942 / train_sft 15968 / eval_sft_epochs 18419·18420 / plot waiter 24076 / monitor 21637），GPU 释放至 0 MiB。
- ep2 提升为最终产物：复制 `checkpoint-126/{adapter_config.json,adapter_model.safetensors,chat_template.jinja}` 到 `outputs/sft_lora_v1/` 根，并写 `SFT_FINAL_SOURCE.txt` 溯源；`checkpoint-126/` 原件（含 optimizer/scheduler/rng/trainer_state）保留可续训。后续 DPO 承接（`train_dpo.py --tag _v1`）默认从该根目录读取。

**2) SFT(ep2) 评测（slug `qwen3.5-2b-hf-sft_v1`，HF 同一推理路径，cublas 12.4.5.8）**
- 产出：`fc_eval/results/qwen3.5-2b-hf-sft_v1/{metrics,scored,raw}`、`fc_eval/results/compare_sft_epochs.md`、`logs/pipeline/sft_curve_v1.png/.csv`（源 checkpoint-126 trainer_state，126 步）。

| 指标(think-off,40例) | base | SFT-ep2 |
| --- | --- | --- |
| 工具选择准确率 | 50.0% | **75.0%** ↑ |
| ask_user 召回率 | 0.0% | **62.5%** ↑（precision 100%，tp5/fp0/fn3） |
| 必填完整率 / 参数值正确率 | 95%/95% | 100%/100% |
| 参数 schema 合法率 | 94.4% | **27.8%** ↓↓ |
| 首步整体正确率 | 45.0% | **22.5%** ↓ |
| 分类首步 A简单 / B多步 / C模糊 | 68.8/37.5/12.5 | **6.2** / 50.0 / 0.0 |

**3) 根因定位（严重格式退化，非评测口径问题）**
- 原始生成实证（A01/A02）：模型选对工具名后，**把该工具（甚至混入其它工具）的所有参数全部输出、值几乎全填字符串 `"None"`**，仅个别真实参数正确（如 `include_item_unique_counts=True`）；部分样例参数重复、且因 `max_new=200` 截断未闭合 `</function></tool_call>`。
- 因“值对、必填齐”故 required_complete/value_correct 仍 100%，但海量 None 占位参数 + 跨工具参数 + 类型不合 → schema_valid 崩塌，进而首步正确率大跌（A 简单最惨 6.2%）。
- 判断：**数据/chat_template 把“未提供的可选参数”序列化成 `None` 占位**，被过拟合放大（loss 早饱和至 ~1e-4 即过拟合信号）。目标 A/B 的“选工具/该问就问”行为已学到，但参数序列化格式坏掉。

**4) 结论与待办**
- ⚠️ **ep2 不可直接作为可用产物**（首步正确率反低于 base）。产物与评测结果已保留用于对比复盘。
- 待办：排查 `src/datagen/*` 与 `src/train/chat_template_train.jinja` 对 `tool_calls.arguments` 的渲染——**只输出已提供参数，禁止 emit `None` 占位、禁止跨工具参数泄漏**；修复后重训（并适当增大评测 `max_new` 防截断）。可另评 ep1(checkpoint-63) 作旁证，但根因属数据/模板、非轮数。

### 2026-07-11 15:05 — schema 暴跌根因定位（**推翻上一条 3)/4) 中"数据/模板 None 占位"的猜测**）

> 对上一条的"根因猜测"做了逐层核验，**证伪了数据/模板假设**，锁定真因为 SFT 过拟合导致的生成退化。

**逐层排除（均已实测）**
- 训练数据 arguments：全库 2220 个 assistant tool_call，**含 null 值的 = 0**；全文件 `"None"` 字符串 = **0**。（那 2220 处 `: null` 全是 tool-call 消息的 `"content": null`，OpenAI 标准，模板按空串处理，与参数无关。）
- 训练模板：用 `chat_template_train.jinja` **实渲染**训练样本，`None` 出现 **0 次**，只按 `tool_call.arguments|items` 渲染已提供参数。→ 训练目标干净。
- 训练 vs 评测 prompt：system prompt **逐字相同**(1328 字)、22 工具集合**完全一致** → 无 prompt 分布漂移。
- base schema 合法率 94.4%（干净）→ 是 **SFT 引入**的退化。

**决定性推理**：训练目标文本**完全不含 `"None"`**（全库+渲染均为 0），监督 loss 只作用于这些目标 token，故 SFT **不可能直接"教会"模型输出 None**；`None` 只能是模型**涌现式退化输出**。

**真因**：**过拟合泛化崩溃（mode collapse）**。loss 早在 epoch≈1.4 塌到 ~1e-4、acc=1（1000 条高度模板化 + all-linear LoRA，严重记忆化）；评测 40 例是刻意去重的 held-out query；模型读到 schema 参数名但丢失"该工具要哪些参数/填什么值"的映射 → 贪心下把参数名全量输出、值统一塌成 `str(None)`，并伴随参数重复/标签不闭合（叠加 `max_new=200` 截断）。

**修正后建议**：① 先评 ep1(checkpoint-63) 旁证逐轮塌陷、且 ep1 可能更优（免重训）；② 重训改为按 N 步 fc_eval 早停、在崩塌前选优，并降轮数/LR/LoRA r 抑制记忆化；③ 评测 `max_new` 调大(如 512)排除截断。数据与模板**无需修改**（本条推翻上一条的数据/模板待办）。

### 2026-07-11 15:20 — ep1 评测结果**推翻上一条"过拟合"判断**，修正真因为暴露偏差/欠收敛

> 采纳建议评测了 ep1(checkpoint-63)，结果与"过拟合塌陷"预测**相反**，据此第二次修正根因。

**三方对比（think-off，40 例，同一 HF 路径，max_new=200）**

| 指标 | base | SFT-ep1 | SFT-ep2 |
|---|---|---|---|
| 首步整体正确率 | 45.0% | 2.5% | 22.5% |
| 工具选择准确率 | 50.0% | 72.5% | 75.0% |
| 参数 schema 合法率 | 94.4% | **5.3%(38)** | **27.8%(36)** |
| ask_user 召回率 | 0.0% | 75.0% | 62.5% |
| A/B/C 首步 | 68.8/37.5/12.5 | 6.2/0/0 | 6.2/50/0 |

**证伪过拟合**：若为过拟合塌陷，训练更少的 ep1 应更好；但 **ep1 明显劣于 ep2**（schema 5.3%→27.8%、首步 2.5%→22.5%），趋势是"训练越多、自由生成越好"，与过拟合相反。ep1 原始输出确认为**同款 None-dump、但更严重**（无参工具 `ugc_get_player_context` 吐 7 个 None 参数）。

**修正真因：暴露偏差（exposure bias）/ free-running 格式欠收敛**（非过拟合、非数据/模板）
- None-dump 从 ep1 即存在且最重，随训练**逐步被纠正**（ep1→ep2 schema +22.5pp）。
- `loss≈0/acc=1` 是 **teacher-forcing** 下的拟合，**不反映自由生成质量**；held-out 贪心解码暴露模型尚未学会"只输出已给参数并及时收尾"，遂枚举 schema 参数名、未提供的填 `str(None)`。
- 结论方向反转：问题是**训练不足/自由生成未收敛**，而非训练过头。**提前停在 ep2 对质量是次优**。

**修正后待办**
1. 从 `checkpoint-126` **resume 续训 ep3~ep5**（optimizer 状态在），逐轮 fc_eval 观察 schema 是否持续爬升——判断能否救活的关键、成本低。
2. 评测 `max_new` 调到 **512** 重跑，排除截断（ep1 输出已 ~354 字符、A01 截断未闭合）。
3. 若续训 schema 仍卡低位，再上针对性手段（增大 LoRA r/LR，或加"仅输出已给参数并收尾"的强化信号）。
- 注：本条推翻上一条(15:05)的"过拟合"结论及其"降轮数/早停"建议；数据与模板仍无需修改。

### 2026-07-11 15:25 — SFT 训练集诊断（结构/覆盖）+ 改进方向（供后续改造） + 从 ep2 续训到 epoch4

**A) 训练集诊断结论：无污染，但有覆盖性缺陷**
- 无直接错误：arguments 含 null=0/2220；全库 `"None"` 字符串=0；调用中 schema 外参数键=0；5 个无参 schema 工具（`ugc_get_player_context` 等）**100% 以空参出现**（`ugc_get_player_context` 201 次全空参）。→ 评测里疯狂吐 None 的正是它，却有 201 条干净示范，**证明 None-dump 非训练集直接所教**。
- `"None"` 来源：**基座模型预训练先验**（Python/API 惯例 可选参→None），SFT 在逐步压制（ep1 5.3%→ep2 27.8%）。

**B) 训练集可行改进方向（后续 SFT 训练集改造依据）**
1. **提高参数个数多样性**：当前每次调用参数数上限仅 **3**（分布 0参18.6%/1参/2参/3参），而模型退化时吐 7~13 参——完全在训练分布外。应加入 **4+ 参数的合法调用**样本，让"长参数列表"进入分布。
2. **补齐可选参数覆盖**：多个可选参数训练中**从未出现**（如 `ugc_get_scene_summary` 的 `include_empty_types/include_item_unique_counts` 覆盖 0；`ugc_get_item_catalog` 的 `category/include_details/surface_type`；`ugc_query_entities_by_area` 的 `limit/level/use_svo_filter`）。应补"正确使用该可选参"与"存在该可选参但正确不填"两类样本，把"可选参数边界"练锐。
3. **强化"只输出已提供参数并及时收尾"的边界**（直接对冲 None-dump 与不闭合）。
- 说明：以上属**增量增强，非推翻重造**；主因仍是暴露偏差/欠收敛+基座先验，数据增强用于**加速收敛、锐化边界**。

**C) 动作：结束 max_new=512 重跑；从 ep2 续训到 epoch4**
- 已停 `max_new=512` 重跑（首例 A01 即 512 token 未闭合→NO_TOOL_CALL，初步印证**截断非主因**；为省时中止，GPU 释放）。
- 备份 ep2：`outputs/sft_lora_v1_ep2_bak/`（因续训会覆盖根产物、`save_total_limit=2` 最终删 checkpoint-126）。
- `train_sft.py` 新增 `--resume`（HF `resume_from_checkpoint`，恢复权重/优化器/调度器/RNG/global_step）。
- 续训命令：`--tag _v1 --epochs 4 --no-eval --resume outputs/sft_lora_v1/checkpoint-126`（从 step126 续到 step252=epoch4，即再训 ep3、ep4）。训练结束根产物=ep4；checkpoint 保留 ep3(189)+ep4(252)（save_total_limit=2）。
- 待办（训练完成后）：评 ep3/ep4，与 base/ep1/ep2 一起看 schema 合法率是否持续爬升（判断"续训能否救活"）。

### 2026-07-11 15:35 — 真 resume 被 torch 版本卡死，改用 `--init-adapter` 续训（修正上一条 C 的续训方式）

- **真 resume 失败**：`--resume outputs/sft_lora_v1/checkpoint-126` 报 `ValueError: ... require torch>=v2.6 ...`（transformers 5.13 的 `_load_optimizer_and_scheduler` → `check_torch_load_is_safe()` 对 `optimizer.pt/scheduler.pt` 的 torch.load 强制要求 torch≥2.6；本环境 torch 2.5.1，safetensors 不受限但优化器状态是 .pt）。升级 torch 风险大（恐破坏 H20/cuBLAS 修复），放弃。
- **改用等价方案**：`train_sft.py` 新增 `--init-adapter`，以 ep2 adapter 作**可训练初始权重**、**fresh 优化器/调度器**继续训练（同 `train_dpo.py` 承接 SFT 的做法，已验证）。仅优化器动量重置（Adam 快速回暖，本轮 LR 从 2e-4 重新退火，反更利于压制 None 先验）。
- **已启动（pid 80768）**：`--tag _v1_ep4 --epochs 2 --no-eval --init-adapter outputs/sft_lora_v1_ep2_bak` → 输出 `outputs/sft_lora_v1_ep4/`。本轮内部 epoch1/2 = **overall epoch3/4**；根产物=overall ep4，`checkpoint-63`=overall ep3（save_total_limit=2 保留 63+126）。总 126 步、ETA ~3.2h；结束自动出曲线 `logs/pipeline/sft_curve_v1_ep4.png/csv`。
- 已确认启动正常：STEP1b 加载 init adapter、train=1000、GPU 55.5GB/100%。
- 待办：训练完成后评 overall-ep3(`outputs/sft_lora_v1_ep4/checkpoint-63`) 与 overall-ep4(`outputs/sft_lora_v1_ep4` 根)，与 base/ep1/ep2 对比 schema 合法率趋势。

### 2026-07-11 15:26 — 工具修复：monitor_train.py 支持 `--log` 指定任意日志

- 现象：`monitor_train.py --watch 10` 无法可视化当前续训。根因：其 `LOGS` 硬编码只认 `chain_sft.log`/`chain_dpo.log`，而续训日志名为 `chain_sft_cont_ep4.log`，monitor 读到的是已 kill 旧 run 的 `chain_sft.log`（冻结在 131/189@14:29）。
- 修复：新增 `--log <文件>`(相对 logs/pipeline 或绝对路径) + `resolve_log()`，覆盖自动选择；`render/save_png/main` 贯穿；文件缺失有提示；向后兼容。
- 实测通过：`monitor_train.py --log chain_sft_cont_ep4.log --watch 10` 正确显示当前续训(step 4/126、loss/acc 实时、GPU 100%)。


### 2026-07-11 18:45 — [自动追加] 五方评测对比结果（base/ep1/ep2/ep3/ep4）

> 本条由 `src/eval/append_compare_to_task.py` 在评测链结束后自动写入（append-only）。
> 数据源：`fc_eval/results/compare_sft_epochs.md`；详细分析/结论待人工补充。

# SFT 逐轮 fc_eval 指标对比（think-off · HF 同一推理路径 · 40 例）

> 用满 1000 条数据训练(--no-eval)，以下游真实指标做逐轮模型选择，替代 eval_loss 早停。


## 一、总体指标

| 指标 | base | SFT-ep1 | SFT-ep2 | SFT-ep3 | SFT-ep4 |
|---|---|---|---|---|---|
| 首步整体正确率 | 45.0% | 2.5% | 22.5% | 0.0% | 0.0% |
| 工具选择准确率 | 50.0% | 72.5% | 75.0% | 22.5% | 17.5% |
| 必填完整率 | 95.0% | 100.0% | 100.0% | 77.8% | 71.4% |
| 参数值正确率 | 95.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 参数schema合法率 | 94.4% | 5.3% | 27.8% | 0.0% | 0.0% |
| ask_user召回率 | 0.0% | 75.0% | 62.5% | 25.0% | 25.0% |
| 工具幻觉率 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 越界/禁用率 | 2.5% | 5.0% | 2.5% | 0.0% | 0.0% |

## 二、分类首步正确率

| 类别 | base | SFT-ep1 | SFT-ep2 | SFT-ep3 | SFT-ep4 |
|---|---|---|---|---|---|
| A 简单单步 | 68.8% | 6.2% | 6.2% | 0.0% | 0.0% |
| B 多步首步 | 37.5% | 0.0% | 50.0% | 0.0% | 0.0% |
| C 模糊追问 | 12.5% | 0.0% | 0.0% | 0.0% | 0.0% |

## 三、选优与过拟合判断

- **推荐 checkpoint：SFT-ep2**（`qwen3.5-2b-hf-sft_v1`），按 首步正确率 > 工具选择 > ask_user 召回 排序选出。
- ⚠️ **过拟合信号**：最后一轮 SFT-ep4 指标低于更早的 SFT-ep2 → 建议采用更早轮次、或减少 epoch 重训。
- 相对 base 提升（SFT-ep2）：首步 -22.5 / 工具选择 +25.0 / ask_user 召回 +62.5。

### 2026-07-11 18:50 — [人工分析] 五方对比结论（补齐上条自动追加）

**曲线呈"倒 V"，峰值在 ep2，ep3/ep4 断崖崩坏**
- ep1→ep2：schema 5.3→27.8、首步 2.5→22.5（改善）；ep2→ep3→ep4：schema 27.8→**0→0**、工具选择 75→22.5→**17.5**、`made_tool_call` 跌到 **25%**（ep4 原始输出 626~664 字符、一路吐 None 到 200 token 不闭合 `</tool_call>`，连合法调用都发不出；比 ep2 的 ~350 字符更失控）。
- 即：**更多 epoch 让 None-dump 显著恶化**，推翻"再多训会救活"的期望。

**重要口径（confound）**：ep3/ep4 非真 resume（torch<2.6 限制），是"ep2 权重初始化 + 全新优化器、LR 从 2e-4 重新退火"。把 2e-4 高 LR 重新砸到已收敛(loss≈0)权重，**极可能是断崖崩坏的主要推手**。故 ep3/ep4 崩坏是"过度训练 + 高 LR 重启"叠加，不能干净归因于轮数；但结论一致：**当前数据下朴素多训只会更糟**。

**本轮目标达成**（验证可行性/排查问题）：① 能正常训练；② "训练后准确率暴跌"问题真实且随训练加剧；③ 最佳可用 checkpoint = **ep2**（已备份 `outputs/sft_lora_v1_ep2_bak`）。

**结论与下一步**
- **冻结 ep2 为当前最佳**（目标 A/B 行为已学到：工具选择 +25、ask_user 召回 +62.5 vs base；代价是 schema/首步）。**不在当前数据上继续加 epoch**。
- 优先做**数据增强**（见 MEMORY「附·临时」与本文件 15:25 条 B）后重训，并**降 LR + fc_eval 逐轮早选**（在 ep2 附近选优，避免冲过头）。
- 可选低优先诊断：**温和低 LR(如 2e-5) 从 ep2 续训**，厘清"轮数 vs LR 重启"的 confound——但因根因属数据/格式，排在数据增强之后。
