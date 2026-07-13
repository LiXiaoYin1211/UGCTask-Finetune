"""
report.py — 读取 results/metrics.json + scored_*.jsonl,生成:
  results/report.md         可读评测报告
  results/chart_overall.png 两模式总体指标对比柱状图
  results/chart_category.png 按类别 first_step_correct 对比
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import results_dir

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MODEL = "qwen3.5:2b"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CAT_LABEL = {"A_simple": "A 简单单步", "B_multistep": "B 多步首步", "C_ambiguous": "C 模糊追问"}


def pct(v):
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        v = v[0]
    return "-" if v is None else "{:.1f}%".format(v)


def first(v):
    return v[0] if isinstance(v, (list, tuple)) else v


def load():
    return json.load(open(os.path.join(RES, "metrics.json"), encoding="utf-8"))


def chart_overall(m):
    modes = [x for x in ("think_off", "think_on") if x in m]
    keys = ["tool_selection", "schema_valid", "required_complete", "value_correct", "first_step_correct"]
    labels = ["工具选择", "schema合法", "必填完整", "参数值正确", "首步整体正确"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(keys))
    w = 0.38
    colors = {"think_off": "#378ADD", "think_on": "#1D9E75"}
    name = {"think_off": "think 关", "think_on": "think 开"}
    for i, mode in enumerate(modes):
        vals = [first(m[mode]["overall"][k]) or 0 for k in keys]
        bars = ax.bar([xi + (i - 0.5) * w for xi in x], vals, w,
                      label=name[mode], color=colors[mode])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, "{:.0f}".format(v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("准确率 %")
    ax.set_title("{} 工具调用能力 · 总体指标(think 开 vs 关)".format(MODEL))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "chart_overall.png"), dpi=130)
    plt.close(fig)


def chart_category(m):
    modes = [x for x in ("think_off", "think_on") if x in m]
    cats = ["A_simple", "B_multistep", "C_ambiguous"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(cats))
    w = 0.38
    colors = {"think_off": "#378ADD", "think_on": "#1D9E75"}
    name = {"think_off": "think 关", "think_on": "think 开"}
    for i, mode in enumerate(modes):
        vals = [first(m[mode]["by_category"].get(c, {}).get("first_step_correct", [0]))
                or 0 for c in cats]
        bars = ax.bar([xi + (i - 0.5) * w for xi in x], vals, w,
                      label=name[mode], color=colors[mode])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, "{:.0f}".format(v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([CAT_LABEL[c] for c in cats])
    ax.set_ylim(0, 105)
    ax.set_ylabel("首步整体正确率 %")
    ax.set_title("按类别拆分 · 首步整体正确率")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "chart_category.png"), dpi=130)
    plt.close(fig)


def md_table_overall(m):
    modes = [x for x in ("think_off", "think_on") if x in m]
    rows = [
        ("工具选择准确率", "tool_selection"),
        ("参数 schema 合法率", "schema_valid"),
        ("必填参数完整率", "required_complete"),
        ("参数值正确率", "value_correct"),
        ("发起工具调用率", "made_tool_call"),
        ("工具幻觉率", "hallucinated_tool"),
        ("命中禁用/越界率", "hit_forbidden"),
        ("**首步整体正确率**", "first_step_correct"),
    ]
    head = "| 指标 | " + " | ".join("think 关" if md == "think_off" else "think 开" for md in modes) + " |"
    sep = "|------|" + "------|" * len(modes)
    lines = [head, sep]
    for label, k in rows:
        cells = []
        for md in modes:
            cells.append(pct(m[md]["overall"][k]))
        lines.append("| {} | {} |".format(label, " | ".join(cells)))
    return "\n".join(lines)


def md_table_category(m):
    modes = [x for x in ("think_off", "think_on") if x in m]
    cats = ["A_simple", "B_multistep", "C_ambiguous"]
    head = "| 类别 | n | " + " | ".join(
        ("首步正确(think关)" if md == "think_off" else "首步正确(think开)") for md in modes) + " |"
    sep = "|------|---|" + "------|" * len(modes)
    lines = [head, sep]
    for c in cats:
        n = first(m[modes[0]]["by_category"].get(c, {}).get("n", 0))
        cells = [pct(m[md]["by_category"].get(c, {}).get("first_step_correct")) for md in modes]
        lines.append("| {} | {} | {} |".format(CAT_LABEL[c], n, " | ".join(cells)))
    return "\n".join(lines)


def md_ask(m):
    modes = [x for x in ("think_off", "think_on") if x in m]
    lines = ["| 模式 | 精确率 | 召回率 | TP | FP | FN |", "|------|--------|--------|----|----|----|"]
    for md in modes:
        a = m[md]["ask_user"]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            "think 关" if md == "think_off" else "think 开",
            pct(a["precision"]), pct(a["recall"]), a["tp"], a["fp"], a["fn"]))
    return "\n".join(lines)


def collect_failures(mode):
    path = os.path.join(RES, "scored_{}.jsonl".format(mode))
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if not r["first_step_correct"]]


def main():
    global RES, MODEL
    args = sys.argv[1:]
    if "--model" in args:
        MODEL = args[args.index("--model") + 1]
    RES = results_dir(MODEL)
    m = load()
    chart_overall(m)
    chart_category(m)

    parts = []
    parts.append("# {} Function-Calling 能力评测报告\n".format(MODEL))
    parts.append("> 模型：本地 Ollama `{}` · 工具表：真实 UGC Runtime 22 工具 · "
                 "数据集：40 例（A简单16 / B多步16 / C模糊8） · 方法：单轮首步动作评测 · temperature=0\n".format(MODEL))
    parts.append("## 一、总体指标（think 开 vs 关）\n")
    parts.append(md_table_overall(m))
    parts.append("\n\n![总体指标](chart_overall.png)\n")
    parts.append("## 二、按类别拆分 · 首步整体正确率\n")
    parts.append(md_table_category(m))
    parts.append("\n\n![按类别](chart_category.png)\n")
    parts.append("## 三、ask_user 该问不问（C 类核心）\n")
    parts.append(md_ask(m))
    parts.append("\n\n## 四、典型失败案例（think 关）\n")
    fails = collect_failures("think_off")
    if fails:
        parts.append("| id | 类别 | 实际首步工具 | 问题 |")
        parts.append("|----|------|------------|------|")
        for r in fails[:25]:
            prob = []
            if not r["made_tool_call"]:
                prob.append("未发起工具调用")
            if r["hit_forbidden"]:
                prob.append("命中禁用工具(跳依赖/破坏性)")
            if r["hallucinated_tool"]:
                prob.append("幻觉工具名")
            if r["made_tool_call"] and not r["tool_selection_correct"] and not r["hit_forbidden"]:
                prob.append("选错工具")
            if r["schema_valid"] is False:
                prob.append("参数schema非法")
            if r["required_complete"] is False:
                prob.append("漏必填参数(如name缺limit)")
            if r["value_correct"] is False:
                prob.append("参数值错")
            parts.append("| {} | {} | {} | {} |".format(
                r["id"], r["category"].split("_")[0],
                r["called_tool"] or "(无)", "；".join(prob) or "其他"))
    else:
        parts.append("（无失败案例）")

    parts.append("\n\n## 五、结论\n")
    of = first(m.get("think_off", {}).get("overall", {}).get("first_step_correct"))
    on = first(m.get("think_on", {}).get("overall", {}).get("first_step_correct"))
    bf = first(m.get("think_off", {}).get("by_category", {}).get("B_multistep", {}).get("first_step_correct"))
    cf = first(m.get("think_off", {}).get("by_category", {}).get("C_ambiguous", {}).get("first_step_correct"))
    parts.append("- 总体首步整体正确率：think 关 **{}**，think 开 **{}**。".format(pct(of), pct(on)))
    parts.append("- 多步任务（B 类，需先查询再操作）首步正确率 **{}**，模糊追问（C 类，需 ask_user）**{}**——"
                 "印证小模型在“依赖前置查询”与“该问不问”上的系统性短板。".format(pct(bf), pct(cf)))
    parts.append("- 该结果与 Berkeley Function Calling Leaderboard 关于“7B 以下小模型工具调用能力骤降”的公开趋势一致，"
                 "为后续 SFT + DPO 后训练提供了可量化的 **基线**。\n")
    md = "\n".join(parts)
    open(os.path.join(RES, "report.md"), "w", encoding="utf-8").write(md)
    print("report written:", os.path.join(RES, "report.md"))


if __name__ == "__main__":
    main()
