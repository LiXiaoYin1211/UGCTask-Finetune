"""train_dpo.py — 在 SFT LoRA 基础上做 QLoRA DPO（8GB / RTX 5060 / 禁 think）。

- 加载 4bit 基座 + SFT 训得的 LoRA 适配器作为起点；DPO 用 peft 时 ref_model=None
  （trl 自动用禁用 adapter 的同一模型作参考，省一份显存，对 8G 关键）。
- max_length 收紧到 768、max_prompt_length 512 防 OOM；beta=0.1。

用法:
  python train_dpo.py                 # 全量
  python train_dpo.py --smoke         # 冒烟:2 步
"""
import argparse
import os
import traceback

# ⚠️ 关键:必须在 import torch 之前先 import datasets/trl,详见 train_sft.py 注释
# (Windows + torch cu128 下 torch 先加载会致 pyarrow DLL access violation 段错误)。
from data_utils import load_dpo          # 内部 import datasets(pyarrow)
from trl import DPOConfig, DPOTrainer    # trl -> datasets -> pyarrow

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
MODEL = os.path.join(ROOT, "..", "model", "Qwen", "Qwen3___5-2B")
SFT_LORA = os.path.join(ROOT, "outputs", "sft_lora")
OUT = os.path.join(ROOT, "outputs", "dpo_lora")
RESULT = os.path.join(ROOT, "logs", "smoke", "smoke_dpo_result.txt")
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
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="")           # 如 "_v2"
    ap.add_argument("--sft", default=None)         # SFT adapter 路径(默认 SFT_LORA+tag)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or (OUT + a.tag)
    sft_dir = a.sft or (SFT_LORA + a.tag)

    if a.smoke and os.path.exists(RESULT):
        open(RESULT, "w", encoding="utf-8").close()
    _log("STEP1 build model+tok ... (tag={} epochs={} beta={})".format(a.tag, a.epochs, a.beta))
    model, tok = build_model_tok()
    _log("STEP1 OK: " + model.__class__.__name__)

    # 承接 SFT 成果:在 SFT 训得的 LoRA adapter 之上继续做 DPO(方案:先SFT→在其上DPO)。
    # 保住 SFT 已改善的必填/参数值,同时用 DPO 专修 ask_user 崩塌与ID混用。
    # is_trainable=True 让该 adapter 可继续训练;ref_model=None 时 trl 用"禁用 adapter 的 base"作参考,省显存。
    if not os.path.isdir(sft_dir):
        raise RuntimeError("未找到 SFT adapter: {}(需先完成 SFT 训练)".format(sft_dir))
    model = PeftModel.from_pretrained(model, sft_dir, is_trainable=True)
    _log("STEP1b loaded SFT adapter (trainable) from " + sft_dir)

    ds = load_dpo(limit=8 if a.smoke else None)
    _log("STEP2 dataset loaded: n=" + str(len(ds)))

    kw = {}
    cfg_params = DPOConfig.__init__.__code__.co_varnames
    if "max_prompt_length" in cfg_params:
        # prompt(system+22工具schema+user) 约 7700 token；DPO 默认对 prompt 做 keep_end 截断,
        # 若 max_prompt_length 过小会把开头的工具 schema 截掉 -> 破坏训推一致。设 7808 覆盖全 prompt。
        kw["max_prompt_length"] = 7808

    cfg = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4 if a.smoke else 16,
        learning_rate=5e-5,
        num_train_epochs=1 if a.smoke else a.epochs,
        max_steps=2 if a.smoke else -1,
        # prompt+完成 最长约 8116；设 8704 覆盖全部(H20 96GB 充裕)。过小会截断 chosen/rejected。
        max_length=8704,
        beta=a.beta,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_strategy="no" if a.smoke else "epoch",
        report_to=[],
        **kw,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,           # adapter 已挂载;ref=None 时 trl 用禁用 adapter 的 base 作参考,省显存
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    _log("STEP3 trainer built, start train ...")
    trainer.train()
    _log("STEP4 train() returned OK")
    if not a.smoke:
        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)
        _log("STEP5 DPO LoRA saved -> " + out_dir)
        print("DPO LoRA saved ->", out_dir)
    else:
        _log("SMOKE OK: DPO 全链路通过")
        print("SMOKE OK: DPO 全链路（加载/数据/前向/反向/step）通过")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log("FAIL: " + type(e).__name__ + ": " + str(e))
        _log(traceback.format_exc())
        raise
