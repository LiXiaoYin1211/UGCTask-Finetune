"""data_utils.py — 加载 finetune/data 下的 SFT/DPO JSONL,适配 trl 1.7 + Qwen3.5 chat_template。

关键适配:
- Qwen3.5 的 chat_template 要求 assistant.tool_calls[].function.arguments 为 **dict**,
  而我们按 OpenAI 标准存为字符串化 JSON → 加载时 parse 回 dict（fix_tool_args）。
- SFT 返回含 "messages" + "tools" 的对话式 Dataset，交给 trl SFTTrainer 自动套模板。
- DPO 返回 {"prompt","chosen","rejected"} 对话式 Dataset。
"""
import copy
import json
import os

from datasets import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
DATA = os.path.join(ROOT, "data")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fix_tool_args(msgs):
    """把 assistant.tool_calls[].function.arguments 从字符串化 JSON 转回 dict(Qwen 模板要求)。"""
    out = copy.deepcopy(msgs)
    for m in out:
        for tc in (m.get("tool_calls") or []):
            a = tc.get("function", {}).get("arguments")
            if isinstance(a, str):
                try:
                    tc["function"]["arguments"] = json.loads(a)
                except Exception:
                    tc["function"]["arguments"] = {}
    return out


def load_sft(path=None, limit=None):
    """返回 Dataset，每条 {"messages":[...], "tools":[...]}（arguments 已转 dict）。"""
    path = path or os.path.join(DATA, "sft.jsonl")
    rows = _read_jsonl(path)
    if limit:
        rows = rows[:limit]
    recs = []
    for r in rows:
        recs.append({"messages": fix_tool_args(r["messages"]),
                     "tools": r.get("tools", [])})
    return Dataset.from_list(recs)


def load_dpo(path=None, limit=None):
    """返回 Dataset，每条 {"prompt":[...], "chosen":[...], "rejected":[...], "tools":[...]}。
    trl 1.7 DPOTrainer 的 tokenize_fn 会读 example["tools"] 并传给 apply_chat_template,
    使 prompt/chosen/rejected 都拼入 <tools> 列表 —— 与 SFT/推理端训推一致(修复 tools 缺失)。"""
    path = path or os.path.join(DATA, "dpo.jsonl")
    rows = _read_jsonl(path)
    if limit:
        rows = rows[:limit]
    recs = []
    for r in rows:
        recs.append({"prompt": fix_tool_args(r["prompt"]),
                     "chosen": fix_tool_args(r["chosen"]),
                     "rejected": fix_tool_args(r["rejected"]),
                     "tools": r.get("tools", [])})
    return Dataset.from_list(recs)


if __name__ == "__main__":
    s = load_sft(limit=5)
    print("SFT sample keys:", s.column_names, "| n:", len(s))
    print("  roles:", [m["role"] for m in s[0]["messages"]])
    d = load_dpo(limit=5)
    print("DPO sample keys:", d.column_names, "| n:", len(d))
    print("  prompt roles:", [m["role"] for m in d[0]["prompt"]])
    print("  chosen roles:", [m["role"] for m in d[0]["chosen"]])
