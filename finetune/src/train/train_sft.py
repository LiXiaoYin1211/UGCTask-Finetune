"""train_sft.py — Qwen3.5-2B QLoRA SFT。

路线：transformers + peft + trl(1.8) 直写。
- 4bit NF4 量化 + bf16 计算；LoRA r=16 / alpha=32 / dropout=0.05 / all-linear；max_length=10240。
- batch=8 + grad_accum=2（等效 batch 16）；gradient_checkpointing 开；paged_adamw_8bit。
- assistant_only_loss=True：只对 assistant 轮算 loss（多步轨迹显式建模），依赖
  chat_template_train.jinja 的 {% generation %} 标记（见 build_model_tok）。
- max_length 必须覆盖完整序列：本数据 system+22 工具 schema 前缀就约 7600 token，轨迹最长 9336；
  过小 + keep_start 截断会把 assistant(label) 全截掉 -> loss 恒 0（详见下方 max_length 注释）。

用法:
  python train_sft.py                  # 全量（默认启用 10% 验证集 + eval_loss 早停）
  python train_sft.py --no-eval        # 关闭验证集/早停，用满 1000 条数据（编排脚本采用此模式）
  python train_sft.py --smoke          # 冒烟:2 步,少量样本,验证全链路
"""
import argparse
import os
import sys
import traceback

# ⚠️ 关键:必须在 import torch 之前先 import datasets/trl(它们会拉起 pyarrow 原生扩展)。
# 在 Windows + torch(cu128) 下,若 torch 先加载,随后 pyarrow 的 DLL 会触发
# access violation 段错误(exit 139),且无任何 Python 异常可捕获。亲测此顺序可规避。
from data_utils import load_sft          # 内部 import datasets(pyarrow)
from trl import SFTConfig, SFTTrainer    # trl -> datasets -> pyarrow
from transformers import EarlyStoppingCallback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
MODEL = os.path.join(ROOT, "..", "model", "Qwen", "Qwen3___5-2B")
# 训练用 chat 模板：在原模板基础上注入 {% generation %} 标记，使 trl 的
# assistant_only_loss 能识别 assistant 生成区（原模板缺该标记，会导致 assistant_masks
# 全 0 -> loss 恒为 0 -> 不学习）。渲染文本与原模板逐字一致，不影响推理/评测。
CHAT_TEMPLATE_TRAIN = os.path.join(HERE, "chat_template_train.jinja")
OUT = os.path.join(ROOT, "outputs", "sft_lora")
RESULT = os.path.join(ROOT, "logs", "smoke", "smoke_sft_result.txt")
os.makedirs(os.path.dirname(RESULT), exist_ok=True)


def _log(msg):
    with open(RESULT, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def build_model_tok():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 覆盖为带 {% generation %} 标记的训练模板（assistant_only_loss 依赖它标记掩码区）。
    with open(CHAT_TEMPLATE_TRAIN, encoding="utf-8") as f:
        tok.chat_template = f.read()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")          # 如 "_v2" -> 输出 outputs/sft_lora_v2
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--no-eval", dest="no_eval", action="store_true",
                    help="关闭验证集/早停，用满 1000 条训练（编排脚本采用此模式）。"
                         "默认(不加此参)会划 10%% 验证集 + eval_loss 早停(patience=1)。"
                         "注：本任务真实目标由 fc_eval 衡量(工具选择/ask_user 等)，eval_loss 仅弱相关，"
                         "过拟合更宜用『逐轮 checkpoint + fc_eval 选优』防线，而非 eval_loss 早停。")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None,
                    help="从指定 checkpoint 目录续训(HF resume_from_checkpoint)，"
                         "恢复权重/优化器/调度器/RNG/global_step；配合更大的 --epochs 可在已训基础上继续。"
                         "注：transformers>=5.13 要求 torch>=2.6 才能 load 优化器状态；torch<2.6 请改用 --init-adapter。")
    ap.add_argument("--init-adapter", dest="init_adapter", default=None,
                    help="以已有 LoRA adapter 作为【可训练初始权重】继续训练(fresh 优化器/调度器)。"
                         "用于 torch<2.6 无法 resume 优化器状态时，从既有 adapter 权重继续训。")
    a = ap.parse_args()
    out_dir = a.out or (OUT + a.tag)

    if a.smoke and os.path.exists(RESULT):
        open(RESULT, "w", encoding="utf-8").close()
    _log("STEP1 build model+tok ... (tag={} epochs={} init_adapter={})".format(a.tag, a.epochs, a.init_adapter))
    model, tok = build_model_tok()
    _log("STEP1 OK: " + model.__class__.__name__)

    if a.init_adapter:
        from peft import PeftModel
        adapter_path = a.init_adapter if os.path.isabs(a.init_adapter) else os.path.join(ROOT, a.init_adapter)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        lora = None  # 已挂载既有 adapter，SFTTrainer 不再重新注入
        _log("STEP1b loaded init adapter (trainable) from " + adapter_path)
    else:
        lora = LoraConfig(
            r=8 if a.smoke else 16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    full = load_sft(limit=16 if a.smoke else None)
    # 划 10% 验证集用于 eval_loss 早停。冒烟或 --no-eval 时不划：用满数据，
    # 过拟合防线改由『逐轮 checkpoint + fc_eval 选优』承担（见 src/eval/compare_epochs.py）。
    if a.smoke or a.no_eval:
        train_ds, eval_ds = full, None
    else:
        sp = full.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = sp["train"], sp["test"]
    _log("STEP2 dataset: train={} eval={}".format(
        len(train_ds), len(eval_ds) if eval_ds is not None else 0))

    do_eval = (not a.smoke) and eval_ds is not None
    cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1 if a.smoke else 8,
        gradient_accumulation_steps=4 if a.smoke else 2,
        learning_rate=2e-4,
        num_train_epochs=1 if a.smoke else a.epochs,
        max_steps=2 if a.smoke else -1,
        # ⚠️ 本数据 system+22工具schema 前缀就约 7600 token，完整轨迹最长 9336。
        # 若 max_length 过小(如4096)+keep_start 截断,assistant(label) 全在 ~7700 之后会被截掉
        # -> labels 全 -100 -> loss 恒为 0(白训)。H20 96GB 充裕,设 10240 覆盖全部序列。
        max_length=10240,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=1,
        eval_strategy="epoch" if do_eval else "no",
        save_strategy="epoch" if do_eval else ("no" if a.smoke else "epoch"),
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to=[],
        assistant_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=1)] if do_eval else []
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=callbacks,
    )
    _log("STEP3 trainer built, start train ... (resume={})".format(a.resume))
    trainer.train(resume_from_checkpoint=a.resume)
    _log("STEP4 train() returned OK")
    if not a.smoke:
        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)
        _log("STEP5 SFT LoRA saved -> " + out_dir)
        print("SFT LoRA saved ->", out_dir)
    else:
        _log("SMOKE OK: SFT 全链路通过")
        print("SMOKE OK: SFT 全链路（加载/数据/前向/反向/step）通过")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log("FAIL: " + type(e).__name__ + ": " + str(e))
        _log(traceback.format_exc())
        raise
