"""compare_three_stage.py — base / SFT / SFT+DPO 三段对比报告(think-off,HF 同一推理路径)。

读取 fc_eval/results/<slug>/metrics.json(由 scorer.py 产出),三个 slug:
  qwen3.5-2b-hf-base / qwen3.5-2b-hf-sft / qwen3.5-2b-hf-dpo
产出:
  fc_eval/results/compare_three_stage.md  (对比表 + 分类拆解 + 结论骨架)
  fc_eval/results/compare_three_stage.png (总体指标三段柱状图)

用法: python compare_three_stage.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
RES = os.path.join(ROOT, "..", "fc_eval", "results")

STAGES = [
    ("base", "qwen3.5-2b-hf-base", "Base(未训练)"),
    ("sft", "qwen3.5-2b-hf-sft", "SFT"),
    ("dpo", "qwen3.5-2b-hf-dpo", "SFT+DPO"),
]
MODE = "think_off"

# 中文字体
for fpath in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fpath).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def load_metrics(slug):
    p = os.path.join(RES, slug, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p, encoding="utf-8"))
    return m.get(MODE)


def num(v):
    """metrics 值形如 [百分比, n](已是百分比)或纯数值;取百分比数字。"""
    if isinstance(v, (list, tuple)):
        return round(v[0], 1) if v and v[0] is not None else None
    if isinstance(v, (int, float)):
        return round(v, 1)
    return None


def g(m, key):
    """从 overall / ask_user 取指标(值已是百分比)。"""
    if not m:
        return None
    ov = m.get("overall", {})
    if key in ov:
        return num(ov[key])
    au = m.get("ask_user", {})
    if key == "ask_user_recall":
        return round(au.get("recall"), 1) if au.get("recall") is not None else None
    if key == "hallucination_rate":
        return num(ov.get("hallucinated_tool"))
    if key == "forbidden_rate":
        return num(ov.get("hit_forbidden"))
    return None


def gcat(m, cat):
    if not m:
        return None
    bc = (m.get("by_category") or {}).get(cat, {})
    return num(bc.get("first_step_correct"))


def main():
    metrics = {k: load_metrics(slug) for k, slug, _ in STAGES}
    have = [k for k in metrics if metrics[k]]
    # 关键指标键(对齐 scorer aggregate 输出)
    KEYS = [
        ("first_step_correct", "首步整体正确率"),
        ("tool_selection", "工具选择准确率"),
        ("schema_valid", "参数schema合法率"),
        ("required_complete", "必填完整率"),
        ("value_correct", "参数值正确率"),
        ("ask_user_recall", "ask_user召回率"),
        ("hallucination_rate", "工具幻觉率"),
        ("forbidden_rate", "越界/禁用率"),
    ]

    parts = []
    parts.append("# Base / SFT / SFT+DPO 三段微调效果对比\n")
    parts.append("> 推理路径:transformers + peft(HF 4bit NF4)统一路径 · think-off · 贪心解码 · "
                 "评测集 fc_eval 40 例(A简单16/B多步16/C模糊8) · 首步动作评测。\n")
    parts.append("> **注**:本表 base 为 **HF 4bit** 跑出的新基线(与 SFT/DPO 同路径,可公平对比);"
                 "与早期 Ollama(Q8_0)基线口径不同,不要混比。\n")

    # 总体对比表
    parts.append("\n## 一、总体指标对比\n")
    header = "| 指标 | " + " | ".join(lbl for k, s, lbl in STAGES if k in have)
    if "base" in have and "dpo" in have:
        header += " | Δ(DPO−Base) |"
    else:
        header += " |"
    parts.append(header)
    parts.append("|" + "---|" * (len(header.split("|")) - 2) + "---|")
    for key, name in KEYS:
        row = "| {} |".format(name)
        vals = {}
        for k, slug, lbl in STAGES:
            if k in have:
                v = g(metrics[k], key)
                vals[k] = v
                row += " {} |".format("—" if v is None else "{:.1f}%".format(v))
        if "base" in have and "dpo" in have and vals.get("base") is not None and vals.get("dpo") is not None:
            d = vals["dpo"] - vals["base"]
            row += " {:+.1f} |".format(d)
        else:
            row += " — |"
        parts.append(row)

    # 分类拆解(first_step_correct by category)
    parts.append("\n## 二、分类首步正确率拆解\n")
    cats = [("A_simple", "A 简单单步"), ("B_multistep", "B 多步首步"), ("C_ambiguous", "C 模糊追问")]
    hdr = "| 类别 | " + " | ".join(lbl for k, s, lbl in STAGES if k in have) + " |"
    parts.append(hdr)
    parts.append("|" + "---|" * (len(hdr.split("|")) - 2) + "---|")
    for ck, cname in cats:
        row = "| {} |".format(cname)
        for k, slug, lbl in STAGES:
            if k in have:
                vv = gcat(metrics[k], ck)
                row += " {} |".format("—" if vv is None else "{:.1f}%".format(vv))
        parts.append(row)

    parts.append("\n## 三、结论\n")
    parts.append("- (训练完成后据实际数据补充:SFT 对工具选择/必填完整的提升、DPO 对 ask_user 的提升。)\n")

    out_md = os.path.join(RES, "compare_three_stage.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("written:", out_md)

    # 图:总体关键指标三段柱状
    plot_keys = [("first_step_correct", "首步整体正确"), ("tool_selection", "工具选择"),
                 ("required_complete", "必填完整"), ("ask_user_recall", "ask_user召回")]
    labels = [n for _, n in plot_keys]
    x = range(len(labels))
    width = 0.26
    colors = {"base": "#888780", "sft": "#378ADD", "dpo": "#7F77DD"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    nstage = len([k for k, s, l in STAGES if k in have])
    offset = -(nstage - 1) / 2.0
    for k, slug, lbl in STAGES:
        if k not in have:
            continue
        vals = [g(metrics[k], key) or 0 for key, _ in plot_keys]
        xs = [i + offset * width for i in x]
        bars = ax.bar(xs, vals, width, label=lbl, color=colors[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, "{:.0f}".format(v),
                    ha="center", va="bottom", fontsize=9)
        offset += 1
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("正确率 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("qwen3.5:2b 微调三段对比(think-off,HF 同一路径)")
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(RES, "compare_three_stage.png")
    fig.savefig(out_png, dpi=130)
    print("written:", out_png)


if __name__ == "__main__":
    main()
