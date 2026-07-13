"""eval_hf.py — 用 transformers+peft 跑 fc_eval 的 40 例首步评测(think-off)。

为什么要它:fc_eval/runner.py 走 Ollama(Q8_0);而训练产出是 HF 4bit LoRA。
要公平对比 base/SFT/DPO,三者必须走**同一条 HF 推理路径**。本脚本即该路径。

关键:
- 复用 fc_eval 的 SYSTEM_PROMPT、tools_ugc.json、dataset.jsonl,保证与既有口径一致。
- Qwen3.5 工具调用是特殊文本格式(非 JSON):
    <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...\n</function>\n</tool_call>
  本脚本解析它,并按工具 schema 把 VALUE 还原成 int/number/bool/对象。
- 输出与 Ollama runner 同构的 raw_think_off.jsonl(parsed.first_tool_call={name,arguments}),
  从而 scorer.py 可零改动复用。
- think-off:add_generation_prompt 已注入空 <think> 块,等价禁 think;贪心解码(temperature=0)。

用法:
  python eval_hf.py --slug qwen3.5-2b-hf-base                 # 纯基座,无 adapter
  python eval_hf.py --slug qwen3.5-2b-hf-sft --adapter outputs/sft_lora
  python eval_hf.py --slug qwen3.5-2b-hf-dpo --adapter outputs/dpo_lora
"""
import argparse
import json
import os
import re
import sys
import time
import traceback

# ⚠️ datasets/transformers 相关在 torch 之前(本脚本不直接 import datasets,但 peft/transformers 链路同理,
# 保持习惯:先不 import torch 的重型依赖。这里仅 transformers+peft+torch,无 datasets,顺序无碍。)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
MODEL = os.path.join(ROOT, "..", "model", "Qwen", "Qwen3___5-2B")
FC = os.path.join(ROOT, "..", "fc_eval")
RESULT_LOG = os.path.join(ROOT, "logs", "pipeline", "eval_hf_progress.txt")
os.makedirs(os.path.dirname(RESULT_LOG), exist_ok=True)

sys.path.insert(0, FC)
from runner import SYSTEM_PROMPT, load_dataset, load_tools  # 复用同一套口径

# 解析 <tool_call> 块的正则
RE_FUNC = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
RE_PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _log(m):
    with open(RESULT_LOG, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")


def _reset_log():
    with open(RESULT_LOG, "w", encoding="utf-8") as f:
        f.write("")


def build_schemas(tools):
    sch = {}
    for t in tools:
        fn = t["function"]
        props = fn.get("parameters", {}).get("properties", {})
        sch[fn["name"]] = props
    return sch


def coerce(val, prop):
    """按 schema 的 type 把字符串 VALUE 还原成 int/number/bool/对象/数组。"""
    val = val.strip()
    typ = (prop or {}).get("type")
    try:
        if typ == "integer":
            return int(val)
        if typ == "number":
            return float(val)
        if typ == "boolean":
            return val.strip().lower() in ("true", "1", "yes")
        if typ in ("object", "array"):
            return json.loads(val)
    except Exception:
        # 退化:尝试 JSON,失败则原样返回字符串
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


def parse_tool_call(text, schemas):
    """从模型输出文本解析首个 <tool_call>。返回 (name, args_dict) 或 (None, {})。"""
    block_start = text.find("<tool_call>")
    if block_start == -1:
        return None, {}
    seg = text[block_start:]
    mfunc = RE_FUNC.search(seg)
    if not mfunc:
        return None, {}
    name = mfunc.group(1).strip()
    body = mfunc.group(2)
    props = schemas.get(name, {})
    args = {}
    for pm in RE_PARAM.finditer(body):
        k = pm.group(1).strip()
        v = pm.group(2)
        args[k] = coerce(v, props.get(k, {}))
    return name, args


class StopOnToolCallEnd(StoppingCriteria):
    """生成出 </tool_call> 即停,首步评测只需第一个工具调用,避免续写浪费。"""
    def __init__(self, tok, prompt_len):
        self.tok = tok
        self.plen = prompt_len

    def __call__(self, input_ids, scores, **kw):
        gen = input_ids[0][self.plen:]
        if gen.shape[0] == 0:
            return False
        # 只解码末尾一小段,检查是否已闭合 tool_call
        tail = self.tok.decode(gen[-12:], skip_special_tokens=True)
        return "</tool_call>" in tail


def build_model_tok(adapter=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def generate(model, tok, tools, query, max_new=200):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}]
    prompt = tok.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    plen = inputs["input_ids"].shape[1]
    stopper = StoppingCriteriaList([StopOnToolCallEnd(tok, plen)])
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id,
                         stopping_criteria=stopper)
    gen = out[0][plen:]
    return tok.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="结果目录名,如 qwen3.5-2b-hf-base")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录(可选)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", dest="max_new", type=int, default=200,
                    help="生成最大新 token 数(默认200; 调大可避免长输出被截断致标签不闭合)")
    a = ap.parse_args()

    if os.path.exists(RESULT_LOG):
        _reset_log()
    adapter = os.path.join(ROOT, a.adapter) if a.adapter else None
    _log("STEP1 load model adapter=%s ..." % adapter)
    model, tok = build_model_tok(adapter)
    _log("STEP1 OK: " + model.__class__.__name__)

    rows = load_dataset()
    if a.limit:
        rows = rows[:a.limit]
    tools = load_tools()
    schemas = build_schemas(tools)

    out_dir = os.path.join(FC, "results", a.slug)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_think_off.jsonl")
    _log("STEP2 start eval n=%d max_new=%d -> %s" % (len(rows), a.max_new, out_path))

    with open(out_path, "w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            t0 = time.time()
            err = None
            try:
                text = generate(model, tok, tools, r["query"], max_new=a.max_new)
                name, args = parse_tool_call(text, schemas)
            except Exception as e:
                text, name, args = "", None, {}
                err = "{}: {}".format(type(e).__name__, e)
            first = None
            if name:
                first = {"name": name, "arguments": args, "args_parse_ok": True}
            rec = {
                "id": r["id"], "category": r["category"], "query": r["query"],
                "mode": "think_off", "latency_s": round(time.time() - t0, 2),
                "error": err,
                "parsed": {"tool_call_count": 1 if name else 0,
                           "first_tool_call": first,
                           "content": text, "thinking": ""},
                "raw_message": {"raw_text": text},
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            _log("[%d/%d] %s -> %s (%.1fs)%s" % (
                i + 1, len(rows), r["id"], name or "NO_TOOL_CALL",
                rec["latency_s"], "  ERR" if err else ""))
    _log("DONE: " + out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log("FAIL: " + type(e).__name__ + ": " + str(e))
        _log(traceback.format_exc())
        raise
