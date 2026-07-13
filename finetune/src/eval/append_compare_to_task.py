"""append_compare_to_task.py — 把五方评测对比结果【自动追加】到 finetune/TASK.md「五、执行日志」。

- append-only：只在文末追加，不改动/覆盖任何历史内容（遵守 MEMORY 维护原则）。
- 内容来自 fc_eval/results/compare_sft_epochs.md（由 compare_epochs.py 生成）。
- 明确标注"[自动追加]"，人工详细分析随后补充。

用法: python append_compare_to_task.py
"""
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # finetune/
TASK = os.path.join(ROOT, "TASK.md")
CMP_MD = os.path.join(ROOT, "..", "fc_eval", "results", "compare_sft_epochs.md")


def main():
    ts = time.strftime("%Y-%m-%d %H:%M")
    if os.path.exists(CMP_MD):
        with open(CMP_MD, encoding="utf-8") as f:
            cmp_body = f.read().strip()
    else:
        cmp_body = "(未找到 compare_sft_epochs.md，评测可能未完成或失败)"

    block = (
        "\n\n### {ts} — [自动追加] 五方评测对比结果（base/ep1/ep2/ep3/ep4）\n\n"
        "> 本条由 `src/eval/append_compare_to_task.py` 在评测链结束后自动写入（append-only）。\n"
        "> 数据源：`fc_eval/results/compare_sft_epochs.md`；详细分析/结论待人工补充。\n\n"
        "{body}\n"
    ).format(ts=ts, body=cmp_body)

    with open(TASK, "a", encoding="utf-8") as f:
        f.write(block)
    print("appended compare result to", TASK, "at", ts)


if __name__ == "__main__":
    main()
