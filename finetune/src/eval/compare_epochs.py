"""compare_epochs.py — SFT 逐轮(checkpoint) fc_eval 指标对比 + 选优/过拟合判断。

用途：在 --no-eval(用满数据)训练下，用『下游 fc_eval 真实指标』做模型选择，替代 eval_loss 早停。
读取 fc_eval/results/<slug>/metrics.json(由 scorer.py 产出) 的 think_off 段，对比若干阶段/轮次。

用法:
  python compare_epochs.py                       # 默认对比 base / SFT-ep2 / SFT-ep3(_v1)
  python compare_epochs.py base:qwen3.5-2b-hf-base ep2:qwen3.5-2b-hf-sft_v1_ep2 ep3:qwen3.5-2b-hf-sft_v1
输出:
  fc_eval/results/compare_sft_epochs.md  (对比表 + 推荐 + 过拟合提示)
  并在 stdout 打印推荐结论。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
RES = os.path.join(ROOT, "..", "fc_eval", "results")
MODE = "think_off"

# 关键指标(键, 中文名)；值均为百分比。
KEYS = [
    ("first_step_correct", "首步整体正确率"),
    ("tool_selection", "工具选择准确率"),
    ("required_complete", "必填完整率"),
    ("value_correct", "参数值正确率"),
    ("schema_valid", "参数schema合法率"),
    ("ask_user_recall", "ask_user召回率"),
    ("hallucinated_tool", "工具幻觉率"),
    ("hit_forbidden", "越界/禁用率"),
]
CATS = [("A_simple", "A 简单单步"), ("B_multistep", "B 多步首步"), ("C_ambiguous", "C 模糊追问")]


def load_metrics(slug):
    p = os.path.join(RES, slug, "metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")).get(MODE)


def num(v):
    if isinstance(v, (list, tuple)):
        return round(v[0], 1) if v and v[0] is not None else None
    if isinstance(v, (int, float)):
        return round(v, 1)
    return None


def get(m, key):
    if not m:
        return None
    ov = m.get("overall", {})
    if key == "ask_user_recall":
        r = (m.get("ask_user") or {}).get("recall")
        return round(r, 1) if r is not None else None
    if key in ov:
        return num(ov[key])
    return None


def gcat(m, cat):
    if not m:
        return None
    return num((m.get("by_category") or {}).get(cat, {}).get("first_step_correct"))


def fmt(v):
    return "—" if v is None else "{:.1f}%".format(v)


def main():
    # 解析 label:slug；默认 base / ep2 / ep3
    args = sys.argv[1:]
    if args:
        stages = []
        for a in args:
            lbl, slug = a.split(":", 1)
            stages.append((lbl, slug))
    else:
        stages = [
            ("base", "qwen3.5-2b-hf-base"),
            ("SFT-ep2", "qwen3.5-2b-hf-sft_v1_ep2"),
            ("SFT-ep3", "qwen3.5-2b-hf-sft_v1"),
        ]
    metrics = {lbl: load_metrics(slug) for lbl, slug in stages}
    have = [(lbl, slug) for lbl, slug in stages if metrics[lbl]]
    if not have:
        print("ERROR: 无任何 metrics.json 可读，检查 slug/评测是否完成。")
        return

    P = []
    P.append("# SFT 逐轮 fc_eval 指标对比（think-off · HF 同一推理路径 · 40 例）\n")
    P.append("> 用满 1000 条数据训练(--no-eval)，以下游真实指标做逐轮模型选择，替代 eval_loss 早停。\n")

    # 总体表
    P.append("\n## 一、总体指标\n")
    hdr = "| 指标 | " + " | ".join(lbl for lbl, _ in have) + " |"
    P.append(hdr)
    P.append("|" + "---|" * (len(have) + 1))
    for key, name in KEYS:
        row = "| {} |".format(name)
        for lbl, _ in have:
            row += " {} |".format(fmt(get(metrics[lbl], key)))
        P.append(row)

    # 分类首步
    P.append("\n## 二、分类首步正确率\n")
    hdr2 = "| 类别 | " + " | ".join(lbl for lbl, _ in have) + " |"
    P.append(hdr2)
    P.append("|" + "---|" * (len(have) + 1))
    for ck, cname in CATS:
        row = "| {} |".format(cname)
        for lbl, _ in have:
            row += " {} |".format(fmt(gcat(metrics[lbl], ck)))
        P.append(row)

    # 选优：只在 SFT 各轮之间挑(排除 base)；主指标 first_step_correct，其次 tool_selection，再 ask_user_recall
    sft_stages = [(lbl, slug) for lbl, slug in have if lbl.lower() != "base"]

    def score_key(lbl):
        m = metrics[lbl]
        return (get(m, "first_step_correct") or -1,
                get(m, "tool_selection") or -1,
                get(m, "ask_user_recall") or -1)

    P.append("\n## 三、选优与过拟合判断\n")
    best_lbl = None
    if sft_stages:
        best_lbl = max((lbl for lbl, _ in sft_stages), key=score_key)
        best_slug = dict(sft_stages)[best_lbl]
        P.append("- **推荐 checkpoint：{}**（`{}`），按 首步正确率 > 工具选择 > ask_user 召回 排序选出。".format(best_lbl, best_slug))
        # 过拟合提示：若最后一轮不如更早的轮
        ordered = [lbl for lbl, _ in sft_stages]
        if len(ordered) >= 2:
            last = ordered[-1]
            prev_best = max(ordered[:-1], key=score_key)
            if score_key(last) < score_key(prev_best):
                P.append("- ⚠️ **过拟合信号**：最后一轮 {} 指标低于更早的 {} → 建议采用更早轮次、或减少 epoch 重训。".format(last, prev_best))
            else:
                P.append("- ✅ 未见明显过拟合：最后一轮不劣于更早轮次。")
    base_m = metrics.get("base")
    if base_m and best_lbl:
        d_fs = (get(metrics[best_lbl], "first_step_correct") or 0) - (get(base_m, "first_step_correct") or 0)
        d_ts = (get(metrics[best_lbl], "tool_selection") or 0) - (get(base_m, "tool_selection") or 0)
        d_au = (get(metrics[best_lbl], "ask_user_recall") or 0) - (get(base_m, "ask_user_recall") or 0)
        P.append("- 相对 base 提升（{}）：首步 {:+.1f} / 工具选择 {:+.1f} / ask_user 召回 {:+.1f}。".format(best_lbl, d_fs, d_ts, d_au))

    out_md = os.path.join(RES, "compare_sft_epochs.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(P) + "\n")
    print("written:", out_md)
    if best_lbl:
        print("推荐 checkpoint:", best_lbl)
    print("\n".join(P[-6:]))


if __name__ == "__main__":
    main()
