"""validate.py — 训练数据质量校验 + 配比/覆盖统计。

校验项:
 1 JSON 结构合法、role 交替合法
 2 tool_call_id 配对(每个 assistant.tool_calls[i].id 有且仅有一条对应 role:tool)
 3 function.arguments 为字符串化合法 JSON
 4 工具名 ∈ 已注册工具集(防幻觉工具,DPO 的 rejected 故意越界除外,单独标注)
 5 ID 二元论:写操作(delete/move/paint)的 unique_id 必须来自前序查询返回,不得等于 item_table_id 编造
   —— 仅对 SFT 正例(chosen 行为)强校验;DPO rejected 故意违规,不校验
 6 与 fc_eval/dataset.jsonl 去重(归一化文本完全相同即判重)
 7 配比/难度/B平衡/DPO矩阵 统计报告
 8 禁 think:不得出现 reasoning_content

用法: python validate.py [--sft data/sft.jsonl] [--dpo data/dpo.jsonl]
"""
import argparse
import json
import os
import re
from collections import Counter

from jsonschema import Draft7Validator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
TOOLS = json.load(open(os.path.join(ROOT, "..", "fc_eval", "tools_ugc.json"), encoding="utf-8"))["tools"]
TOOL_SCHEMAS = {t["function"]["name"]: t["function"].get("parameters", {}) for t in TOOLS}
VALID_TOOLS = set(TOOL_SCHEMAS)
WRITE_TOOLS = {"ugc_delete_entity", "ugc_move_entity", "ugc_paint_color_surface",
               "ugc_paint_texture_surface", "ugc_place_on_surface"}

EVAL_PATH = os.path.join(ROOT, "..", "fc_eval", "dataset.jsonl")


def norm(s):
    return re.sub(r"\s+", "", (s or "").lower())


def eval_queries():
    qs = set()
    if os.path.exists(EVAL_PATH):
        for line in open(EVAL_PATH, encoding="utf-8"):
            line = line.strip()
            if line:
                qs.add(norm(json.loads(line)["query"]))
    return qs


def check_messages(msgs, errors, where, allow_oob=False, strict_id=True, allow_trailing_call=False):
    """校验一段 messages 序列;返回该段里出现的工具名集合。
    allow_trailing_call: DPO completion 末尾允许一个未配对的 tool_call(决策终点),不算错。"""
    used = set()
    pending = {}  # call_id -> tool_name 等待对应 tool 消息
    pending_order = []  # 记录 call_id 出现顺序,用于豁免末尾未配对
    queried_uids = set()
    known_item_ids = set()
    for m in msgs:
        role = m.get("role")
        if "reasoning_content" in m:
            errors.append("{}: 出现 reasoning_content(禁 think)".format(where))
        if role == "assistant" and m.get("tool_calls"):
            if m.get("content") not in (None, ""):
                # content 与 tool_calls 并存是允许的,但本数据约定纯调用时 content=None
                pass
            for tc in m["tool_calls"]:
                name = tc["function"]["name"]
                used.add(name)
                # arguments 必须是字符串化 JSON
                raw = tc["function"]["arguments"]
                if not isinstance(raw, str):
                    errors.append("{}: arguments 非字符串({})".format(where, name))
                    args = {}
                else:
                    try:
                        args = json.loads(raw)
                    except Exception:
                        errors.append("{}: arguments 非法 JSON({})".format(where, name))
                        args = {}
                # 工具名合法性
                if name not in VALID_TOOLS:
                    if not allow_oob:
                        errors.append("{}: 调用未注册工具 {}".format(where, name))
                else:
                    # schema 校验
                    try:
                        Draft7Validator(TOOL_SCHEMAS[name]).validate(args)
                    except Exception as e:
                        if not allow_oob:
                            errors.append("{}: schema 不合法 {} ({})".format(where, name, str(e)[:60]))
                # ID 二元论:写操作的 unique_id 不得等于已知 item_table_id 且应来自查询
                if strict_id and name in WRITE_TOOLS | {"ugc_get_entity_detail", "ugc_query_paint_surfaces"}:
                    uid = args.get("unique_id") or args.get("target_entity_id")
                    if uid is not None:
                        if uid in known_item_ids:
                            errors.append("{}: 疑似 ID 混用,操作用了 item_table_id={}".format(where, uid))
                        elif queried_uids and uid not in queried_uids:
                            errors.append("{}: 使用了未经查询返回的 unique_id={}".format(where, uid))
                pending[tc["id"]] = name
                pending_order.append(tc["id"])
        elif role == "tool":
            cid = m.get("tool_call_id")
            if cid not in pending:
                errors.append("{}: tool 消息无配对 tool_call_id={}".format(where, cid))
            else:
                # 记录查询返回的 unique_id / item_table_id(供后续 ID 校验)
                try:
                    res = json.loads(m["content"])
                    for e in res.get("entities", []):
                        if "unique_id" in e:
                            queried_uids.add(e["unique_id"])
                    for it in res.get("items", []):
                        known_item_ids.add(it.get("item_table_id"))
                    for it in res.get("summary", []):
                        known_item_ids.add(it.get("item_table_id"))
                except Exception:
                    pass
                del pending[cid]
    # 末尾未配对豁免:DPO completion 最后一个 tool_call 是决策终点,无需 tool 结果
    if pending and allow_trailing_call:
        last_id = pending_order[-1]
        if last_id in pending and len(pending) == 1:
            del pending[last_id]
    if pending:
        errors.append("{}: 有 tool_call 未配对 tool 结果: {}".format(where, list(pending.values())))
    return used


def validate_sft(path):
    errors = []
    evals = eval_queries()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    cat = Counter()
    diff = Counter()
    bsub = Counter()
    tool_cov = Counter()
    dup = 0
    for r in rows:
        wid = r.get("id", "?")
        meta = r.get("meta", {})
        cat[meta.get("category")] += 1
        if meta.get("category") == "A":
            diff[meta.get("difficulty")] += 1
        if meta.get("category") == "B":
            bsub[meta.get("subtype")] += 1
        if norm(meta.get("query")) in evals:
            dup += 1
            errors.append("{}: query 与评测集重复: {}".format(wid, meta.get("query")))
        used = check_messages(r["messages"], errors, wid, allow_oob=False, strict_id=True)
        for t in used:
            tool_cov[t] += 1
        # 「不多不漏」硬校验:实际调用的工具集合必须与 meta.tools_used 声明完全一致。
        # 多调 -> 出现声明外的工具;漏调 -> 声明的工具没调到。二者都报错。
        declared = set(meta.get("tools_used") or [])
        if declared or used:
            extra = used - declared
            missing = declared - used
            if extra:
                errors.append("{}: 多调工具(声明外) {}".format(wid, sorted(extra)))
            if missing:
                errors.append("{}: 漏调工具(声明未出现) {}".format(wid, sorted(missing)))
    # 全工具覆盖硬校验:任一已注册工具 0 覆盖即报错(治 paint_surfaces 类长尾漏训 -> schema 幻觉)
    MIN_COV = 5
    missing = sorted(t for t in VALID_TOOLS if tool_cov.get(t, 0) == 0)
    low = sorted((t, tool_cov[t]) for t in VALID_TOOLS if 0 < tool_cov.get(t, 0) < MIN_COV)
    if missing:
        errors.append("工具覆盖缺口: {} 个工具 0 覆盖 -> {}".format(len(missing), missing))
    if low:
        errors.append("工具覆盖偏低(<{}): {}".format(MIN_COV, low))
    return {"n": len(rows), "errors": errors, "cat": cat, "diff": diff,
            "bsub": bsub, "tool_cov": tool_cov, "dup": dup,
            "missing_tools": missing, "low_tools": low}


def validate_dpo(path):
    errors = []
    evals = eval_queries()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    types = Counter()
    dup = 0
    for r in rows:
        wid = r.get("id", "?")
        meta = r.get("meta", {})
        types[meta.get("type")] += 1
        if norm(meta.get("query")) in evals:
            dup += 1
            errors.append("{}: query 与评测集重复: {}".format(wid, meta.get("query")))
        # chosen 严格校验; rejected 允许越界/违规(故意的负例)
        chosen_used = check_messages(r["prompt"] + r["chosen"], errors, wid + ".chosen",
                                     allow_oob=False, strict_id=True, allow_trailing_call=True)
        check_messages(r["prompt"] + r["rejected"], errors, wid + ".rejected",
                       allow_oob=True, strict_id=False, allow_trailing_call=True)
        # 「不多不漏」:chosen 实际工具须与 meta.chosen_tools 声明一致(仅在声明了时校验)
        if "chosen_tools" in meta:
            declared = set(meta["chosen_tools"])
            extra = chosen_used - declared
            missing = declared - chosen_used
            if extra:
                errors.append("{}.chosen: 多调工具 {}".format(wid, sorted(extra)))
            if missing:
                errors.append("{}.chosen: 漏调工具 {}".format(wid, sorted(missing)))
    return {"n": len(rows), "errors": errors, "types": types, "dup": dup}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=os.path.join(ROOT, "data", "sft.jsonl"))
    ap.add_argument("--dpo", default=os.path.join(ROOT, "data", "dpo.jsonl"))
    ap.add_argument("--stats", default="", help="若指定,写 Markdown 统计报告到该路径")
    a = ap.parse_args()

    sft_r = validate_sft(a.sft) if os.path.exists(a.sft) else None
    dpo_r = validate_dpo(a.dpo) if os.path.exists(a.dpo) else None

    print("=" * 56)
    if sft_r:
        r = sft_r
        print("SFT n={} 重复={} 错误={}".format(r["n"], r["dup"], len(r["errors"])))
        print("  类别配比:", dict(r["cat"]))
        print("  A难度:", dict(r["diff"]))
        print("  B子类(该问/不该问):", dict(r["bsub"]))
        print("  工具覆盖:", dict(sorted(r["tool_cov"].items(), key=lambda x: x[1])))
        for e in r["errors"][:15]:
            print("   ! ", e)
    print("-" * 56)
    if dpo_r:
        r = dpo_r
        print("DPO n={} 重复={} 错误={}".format(r["n"], r["dup"], len(r["errors"])))
        total = sum(r["types"].values())
        print("  矩阵占比:", {k: "{:.0%}".format(v / total) for k, v in r["types"].items()})
        for e in r["errors"][:15]:
            print("   ! ", e)
    print("=" * 56)

    if a.stats:
        _write_stats(a.stats, sft_r, dpo_r)
        print("stats ->", a.stats)


def _write_stats(path, sft_r, dpo_r):
    L = ["# 训练数据统计报告（validate.py 自动生成）\n"]
    if sft_r:
        r = sft_r
        ok = "通过" if (r["dup"] == 0 and len(r["errors"]) == 0) else "未通过"
        L.append("## SFT（{} 条）— 校验{}\n".format(r["n"], ok))
        L.append("| 项 | 值 |")
        L.append("|----|----|")
        L.append("| 总数 | {} |".format(r["n"]))
        L.append("| 与评测集重复 | {} |".format(r["dup"]))
        L.append("| 校验错误 | {} |".format(len(r["errors"])))
        L.append("| 类别配比 | {} |".format(dict(r["cat"])))
        L.append("| A 内难度 | {} |".format(dict(r["diff"])))
        L.append("| B 子类(该问:不该问) | {} |".format(dict(r["bsub"])))
        L.append("\n### 工具覆盖（升序）\n")
        L.append("| 工具 | 出现轨迹数 |")
        L.append("|------|-----------|")
        for t, c in sorted(r["tool_cov"].items(), key=lambda x: x[1]):
            L.append("| {} | {} |".format(t, c))
        L.append("")
    if dpo_r:
        r = dpo_r
        ok = "通过" if (r["dup"] == 0 and len(r["errors"]) == 0) else "未通过"
        total = sum(r["types"].values())
        L.append("## DPO（{} 对）— 校验{}\n".format(r["n"], ok))
        L.append("| 失败类型 | 对数 | 占比 |")
        L.append("|----------|------|------|")
        for k, v in r["types"].items():
            L.append("| {} | {} | {:.0%} |".format(k, v, v / total))
        L.append("| 与评测集重复 | {} | - |".format(r["dup"]))
        L.append("| 校验错误 | {} | - |".format(len(r["errors"])))
        L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
