"""
compare.py — 跨模型总体指标对比(base vs cand)。

读取 results/<slug>/metrics.json,产出:
  results/compare_<base>_vs_<cand>.md   对比表(含 Δ 列)
  results/compare_overall.png           分组对比柱状图(think_off / think_on 两子图)

用法:
  <venv_python> compare.py [--base qwen3.5:2b] [--cand qwen3.5:9b]
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import results_dir, model_slug

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# (key, 中文标签, 方向)  direction: +1 越高越好, -1 越低越好
METRICS = [
    ("tool_selection", "工具选择", +1),
    ("schema_valid", "schema合法", +1),
    ("required_complete", "必填完整", +1),
    ("value_correct", "参数值正确", +1),
    ("made_tool_call", "发起调用", +1),
    ("hallucinated_tool", "工具幻觉", -1),
    ("hit_forbidden", "命中禁用/越界", -1),
    ("first_step_correct", "首步整体正确", +1),
]
# 正向指标(画图用)
POS_KEYS = [m for m in METRICS if m[2] == +1]


def load_metrics(model):
    path = os.path.join(results_dir(model), "metrics.json")
    if not os.path.exists(path):
        raise SystemExit("缺少 metrics: {} (请先对 {} 跑 scorer.py)".format(path, model))
    return json.load(open(path, encoding="utf-8"))


def cell(m, mode, key):
    v = m.get(mode, {}).get("overall", {}).get(key)
    if isinstance(v, (list, tuple)):
        return v[0]
    return v


def fmt(v):
    return "-" if v is None else "{:.1f}".format(v)


def delta_str(base, cand, direction):
    if base is None or cand is None:
        return "-"
    d = cand - base
    arrow = ""
    if abs(d) >= 0.05:
        good = (d > 0 and direction > 0) or (d < 0 and direction < 0)
        arrow = " ↑" if d > 0 else " ↓"
        arrow += "(优)" if good else "(劣)"
    return "{:+.1f}{}".format(d, arrow)


def build_table(mb, mc, base, cand):
    lines = []
    lines.append("| 指标 | {b}(关) | {c}(关) | Δ关 | {b}(开) | {c}(开) | Δ开 |".format(b=base, c=cand))
    lines.append("|------|--------|--------|------|--------|--------|------|")
    for key, label, direction in METRICS:
        bo = cell(mb, "think_off", key)
        co = cell(mc, "think_off", key)
        bn = cell(mb, "think_on", key)
        cn = cell(mc, "think_on", key)
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            label, fmt(bo), fmt(co), delta_str(bo, co, direction),
            fmt(bn), fmt(cn), delta_str(bn, cn, direction)))
    return "\n".join(lines)


def chart(mb, mc, base, cand):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    labels = [lab for _, lab, _ in POS_KEYS]
    keys = [k for k, _, _ in POS_KEYS]
    for ax, mode, title in zip(axes, ("think_off", "think_on"),
                               ("think 关", "think 开")):
        x = range(len(keys))
        w = 0.38
        bvals = [cell(mb, mode, k) or 0 for k in keys]
        cvals = [cell(mc, mode, k) or 0 for k in keys]
        b1 = ax.bar([xi - w / 2 for xi in x], bvals, w, label=base, color="#378ADD")
        b2 = ax.bar([xi + w / 2 for xi in x], cvals, w, label=cand, color="#D85A30")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                        "{:.0f}".format(b.get_height()), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 108)
        ax.set_title("{} · 总体正向指标".format(title))
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("准确率 %")
    fig.suptitle("{} vs {} · 工具调用能力对比".format(base, cand), fontsize=13)
    fig.tight_layout()
    out = os.path.join(RES, "compare_overall.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def metric_glossary():
    """指标含义说明(固定内容,随报告生成,保证可复现)。"""
    lines = []
    lines.append("## 〇、指标含义说明\n")
    lines.append("> 评测方法:单轮「首步动作」评测——给定 `system + tools + 用户query`,只看模型的**第一个动作**是否正确。"
                 "以下所有指标均针对该首步,取值为总体(overall)百分比。\n")
    lines.append("| 指标 | 定义 | 判正/判误 | 计入范围 |")
    lines.append("|------|------|----------|----------|")
    lines.append("| **工具选择** | 首步实际调用的工具名是否落在该用例预标注的 `acceptable_tools` 集合内 | "
                 "命中=正确;调集合外工具、或C类该追问却没调 `ask_user`=错误。**只看选没选对工具,不管参数** | 全部 40 例 |")
    lines.append("| **schema合法** | 模型所填参数(`arguments`)能否通过该工具的 JSON Schema 校验 | "
                 "类型对、无非法字段、枚举合规=合法。属**语法/类型层** | 仅\"已发起且工具存在\"的样本 |")
    lines.append("| **必填完整** | 完成该任务在**语义上必须**提供的参数(数据集标注的 `must_have_keys`)是否齐全 | "
                 "全部出现=完整;漏任一=不全。如 catalog 给了 `name` 却漏 `limit`(条件必填) | 仅\"工具选对\"的样本 |")
    lines.append("| **参数值正确** | 对有确定答案的参数(`value_constraints`),模型填的**值**是否精确匹配 | "
                 "如\"id 2002\"必须真填 2002,填别的=错。属**取值精确层** | 仅\"工具选对\"的样本 |")
    lines.append("| **发起调用** | 首步是否真的发起了工具调用(而非纯文本回答) | 有调用=是。"
                 "注意:**非越高越好**——信息严重不足时硬调工具反而是\"鲁莽\",克制不调可能更优 | 全部 40 例 |")
    lines.append("| **工具幻觉** | 是否调用了不存在的工具名 | 调了未注册工具=幻觉(越低越好) | 全部 40 例 |")
    lines.append("| **命中禁用/越界** | 是否调用了该用例 `forbidden_tools`(跳过前置依赖、破坏性误操作、越 scope) | 命中=越界(越低越好) | 全部 40 例 |")
    lines.append("| **首步整体正确** | 综合端到端判定 | **同时满足**:①工具选对 ②未命中禁用 ③无工具幻觉 "
                 "④schema 不为非法 ⑤必填不缺、参数值不错——全对才算成功 | 全部 40 例 |")
    lines.append("\n**三个参数指标的层级关系(粒度由粗到细,互不冲突):** "
                 "`schema合法`(语法/类型合规) → `必填完整`(任务必需参数齐全) → `参数值正确`(填的值精确)。"
                 "例:某次调用 JSON 完全合法(schema✓),但漏了任务需要的 `limit`(必填✗)——二者各查一层,并不矛盾。\n")
    lines.append("> 任务分类:**A 简单单步**(一句话映射到单一工具)、**B 多步首步**(需先查询拿 unique_id 再操作,只评第一步是否为正确前置查询)、"
                 "**C 模糊追问**(信息不足,期望触发 `ask_user` 而非瞎猜执行),配比 16/16/8。\n")
    return "\n".join(lines)


def ask_block(mb, mc, base, cand):
    lines = ["| 模型 | think | ask_user 召回 | TP | FN |", "|------|-------|--------------|----|----|"]
    for tag, m in ((base, mb), (cand, mc)):
        for mode in ("think_off", "think_on"):
            a = m.get(mode, {}).get("ask_user", {})
            r = a.get("recall")
            lines.append("| {} | {} | {} | {} | {} |".format(
                tag, "关" if mode == "think_off" else "开",
                "-" if r is None else "{:.1f}%".format(r), a.get("tp", "-"), a.get("fn", "-")))
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    base, cand = "qwen3.5:2b", "qwen3.5:9b"
    if "--base" in args:
        base = args[args.index("--base") + 1]
    if "--cand" in args:
        cand = args[args.index("--cand") + 1]
    mb, mc = load_metrics(base), load_metrics(cand)

    chart_path = chart(mb, mc, base, cand)

    of_b = cell(mb, "think_off", "first_step_correct")
    of_c = cell(mc, "think_off", "first_step_correct")
    on_b = cell(mb, "think_on", "first_step_correct")
    on_c = cell(mc, "think_on", "first_step_correct")

    parts = []
    parts.append("# 模型尺寸对照：{} vs {}\n".format(base, cand))
    parts.append("> 控制变量实验：相同工具表(22 工具) + 相同数据集(40 例) + 相同打分逻辑，"
                 "唯一变量为模型尺寸。指标为总体(overall)首步动作正确率，单位 %。\n")
    parts.append(metric_glossary())
    parts.append("## 一、总体指标对比（Δ = {} − {}，正向指标越高越好，幻觉/越界越低越好）\n".format(cand, base))
    parts.append(build_table(mb, mc, base, cand))
    parts.append("\n\n![对比图](compare_overall.png)\n")
    parts.append("## 二、ask_user 该问不问（C 类核心短板）\n")
    parts.append(ask_block(mb, mc, base, cand))
    parts.append("\n\n## 三、结论摘要\n")
    parts.append("- 首步整体正确率(think 关)：{} **{}** → {} **{}**（Δ {}）。".format(
        base, fmt(of_b), cand, fmt(of_c), delta_str(of_b, of_c, +1)))
    parts.append("- 首步整体正确率(think 开)：{} **{}** → {} **{}**（Δ {}）。".format(
        base, fmt(on_b), cand, fmt(on_c), delta_str(on_b, on_c, +1)))
    parts.append("- 关键判断：对比 {} 与 {} 在 **ask_user 召回** 与 **多步前置查询** 上的差异，"
                 "可区分“短板源于模型容量(尺寸越大越好)”还是“小模型固有缺陷(需后训练修复)”。\n".format(base, cand))
    md = "\n".join(parts)
    out_md = os.path.join(RES, "compare_{}_vs_{}.md".format(model_slug(base), model_slug(cand)))
    open(out_md, "w", encoding="utf-8").write(md)
    print("written:", out_md)
    print("written:", chart_path)


if __name__ == "__main__":
    main()
