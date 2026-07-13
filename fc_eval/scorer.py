"""
scorer.py — 对 results/raw_<mode>.jsonl 打分,产出 results/scored_<mode>.jsonl 与
results/metrics.json(两模式指标汇总)。

打分维度(对齐 plan):
 1 工具选择准确率(tool_selection)        首步工具 ∈ acceptable_tools
 2 参数 schema 合法率(schema_valid)       arguments 通过该工具 JSON Schema
 3 必填参数完整率(required_complete)      must_have_keys 全部出现
 4 参数值正确率(value_correct)            value_constraints 全部匹配
 5 工具/字段幻觉率(hallucination)         调用了不存在的工具,或 forbidden_tools
 6 ask_user 精确/召回(C类该问不问)
 7 越界调用率(out_of_scope / forbidden)   命中 forbidden_tools
 8 整体首步正确(first_step_correct)       综合判定(见下)
"""
import json
import os
import sys
from jsonschema import Draft7Validator

from common import results_dir

HERE = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_truth():
    rows = load_jsonl(os.path.join(HERE, "dataset.jsonl"))
    return {r["id"]: r for r in rows}


def load_tool_schemas():
    tools = json.load(open(os.path.join(HERE, "tools_ugc.json"), encoding="utf-8"))["tools"]
    valid_names = set()
    schemas = {}
    for t in tools:
        fn = t["function"]
        valid_names.add(fn["name"])
        schemas[fn["name"]] = fn.get("parameters", {})
    return valid_names, schemas


def score_one(rec, truth, valid_names, schemas):
    """对单条响应打分,返回 dict of bool/None 指标。"""
    gt = truth[rec["id"]]
    parsed = rec.get("parsed") or {}
    ftc = parsed.get("first_tool_call") or {}
    called = ftc.get("name")
    args = ftc.get("arguments") if isinstance(ftc.get("arguments"), dict) else {}
    is_C = gt["category"] == "C_ambiguous"

    out = {
        "id": rec["id"], "category": gt["category"], "mode": rec["mode"],
        "called_tool": called, "error": rec.get("error"),
        "made_tool_call": bool(called),
    }

    # 幻觉:调用了不存在的工具
    out["hallucinated_tool"] = bool(called) and (called not in valid_names)
    # 越界/禁用:命中 forbidden_tools
    out["hit_forbidden"] = bool(called) and (called in gt.get("forbidden_tools", []))

    # 工具选择准确(仅当模型确实发起调用)
    acceptable = set(gt.get("acceptable_tools", []))
    if called is None:
        out["tool_selection_correct"] = (not is_C) and False  # C类不调工具也算错(应调ask_user)
        # 对C类,没调任何工具=既没问也没做=错
        out["tool_selection_correct"] = False
    else:
        out["tool_selection_correct"] = called in acceptable

    # schema 合法率(仅对有 schema 的真实工具)
    if called and called in schemas:
        try:
            Draft7Validator(schemas[called]).validate(args)
            out["schema_valid"] = True
        except Exception:
            out["schema_valid"] = False
    else:
        out["schema_valid"] = None  # 无调用或幻觉工具,不计入

    # arguments 解析是否成功(畸形JSON)
    out["args_parse_ok"] = ftc.get("args_parse_ok") if ftc else None

    # 必填参数完整(仅当工具选对时才有意义)
    must = gt.get("expected_args", {}).get("must_have_keys", [])
    if out["tool_selection_correct"] and must:
        out["required_complete"] = all(k in args for k in must)
    elif out["tool_selection_correct"]:
        out["required_complete"] = True  # 无要求即视为满足
    else:
        out["required_complete"] = None

    # 参数值正确
    vc = gt.get("expected_args", {}).get("value_constraints", {})
    if out["tool_selection_correct"] and vc:
        out["value_correct"] = all(str(args.get(k)) == str(v) for k, v in vc.items())
    elif out["tool_selection_correct"]:
        out["value_correct"] = True
    else:
        out["value_correct"] = None

    # ask_user 判定(仅C类)
    if is_C:
        out["asked_user"] = (called == "ask_user")
        out["should_ask"] = True
    else:
        out["asked_user"] = (called == "ask_user")
        out["should_ask"] = False

    # 整体首步正确:选对工具 + (若有)schema合法 + 必填齐 + 值对 + 未命中禁用
    fs = out["tool_selection_correct"] and not out["hit_forbidden"] and not out["hallucinated_tool"]
    if fs and out["schema_valid"] is False:
        fs = False
    if fs and out["required_complete"] is False:
        fs = False
    if fs and out["value_correct"] is False:
        fs = False
    out["first_step_correct"] = bool(fs)
    return out


def rate(vals):
    """对 list of bool/None 求比率,忽略 None。返回 (pct, n)。"""
    xs = [v for v in vals if v is not None]
    if not xs:
        return None, 0
    return round(100.0 * sum(1 for v in xs if v) / len(xs), 1), len(xs)


def aggregate(scored):
    cats = ["A_simple", "B_multistep", "C_ambiguous"]
    agg = {"overall": {}, "by_category": {}}

    def block(rows):
        d = {}
        d["n"] = len(rows)
        d["tool_selection"] = rate([r["tool_selection_correct"] for r in rows])
        d["schema_valid"] = rate([r["schema_valid"] for r in rows])
        d["required_complete"] = rate([r["required_complete"] for r in rows])
        d["value_correct"] = rate([r["value_correct"] for r in rows])
        d["first_step_correct"] = rate([r["first_step_correct"] for r in rows])
        d["made_tool_call"] = rate([r["made_tool_call"] for r in rows])
        d["hallucinated_tool"] = rate([r["hallucinated_tool"] for r in rows])
        d["hit_forbidden"] = rate([r["hit_forbidden"] for r in rows])
        return d

    agg["overall"] = block(scored)
    for c in cats:
        rows = [r for r in scored if r["category"] == c]
        if rows:
            agg["by_category"][c] = block(rows)

    # ask_user 精确/召回(全集)
    tp = sum(1 for r in scored if r["should_ask"] and r["asked_user"])
    fp = sum(1 for r in scored if (not r["should_ask"]) and r["asked_user"])
    fn = sum(1 for r in scored if r["should_ask"] and (not r["asked_user"]))
    prec = round(100.0 * tp / (tp + fp), 1) if (tp + fp) else None
    rec = round(100.0 * tp / (tp + fn), 1) if (tp + fn) else None
    agg["ask_user"] = {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
    return agg


def main():
    args = sys.argv[1:]
    model = "qwen3.5:2b"
    suffix = ""
    if "--model" in args:
        model = args[args.index("--model") + 1]
    if "--suffix" in args:
        suffix = args[args.index("--suffix") + 1]
    rd = results_dir(model, ensure=True, suffix=suffix)
    truth = load_truth()
    valid_names, schemas = load_tool_schemas()
    metrics = {}
    for mode in ("think_off", "think_on"):
        raw_path = os.path.join(rd, "raw_{}.jsonl".format(mode))
        if not os.path.exists(raw_path):
            continue
        raw = load_jsonl(raw_path)
        scored = [score_one(r, truth, valid_names, schemas) for r in raw]
        with open(os.path.join(rd, "scored_{}.jsonl".format(mode)),
                  "w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        metrics[mode] = aggregate(scored)
    json.dump(metrics, open(os.path.join(rd, "metrics.json"),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[{}] scored ->".format(model), rd)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
