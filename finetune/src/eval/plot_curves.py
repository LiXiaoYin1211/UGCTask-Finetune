"""plot_curves.py — 训练曲线持久化(PNG + CSV)，SFT / DPO 通用。

数据来源优先级：
  1) trainer_state.json 的 log_history(权威：干净浮点、含 eval_loss 若有) —
     取 outputs/{sft,dpo}_lora<tag>/(checkpoint-* 最新 或 根目录)。
  2) 回退：解析 logs/pipeline/chain_{stage}.log(tqdm 文本)。
产出：
  - PNG：loss(+SFT: mean_token_accuracy / DPO: rewards/accuracies + margins)，含 eval_loss 散点(若有)。
  - CSV：完整 log_history 序列，便于后续复绘/分析。

用法:
  python plot_curves.py --stage sft --tag _v1
  python plot_curves.py --stage dpo --tag _v1 --out xxx.png --csv xxx.csv
  python plot_curves.py --stage auto            # 按 outputs/日志新鲜度自动判定
"""
import argparse
import csv as csvmod
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/
OUTPUTS = os.path.join(ROOT, "outputs")
LOGDIR = os.path.join(ROOT, "logs", "pipeline")
LOGS = {"sft": os.path.join(LOGDIR, "chain_sft.log"), "dpo": os.path.join(LOGDIR, "chain_dpo.log")}
RE_BLOCK = re.compile(r"\{[^{}]*'loss'[^{}]*\}")


def _kv(block, key):
    m = re.search(r"'%s':\s*'?(-?[\d.eE+]+)'?" % re.escape(key), block)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def find_trainer_state(stage, tag):
    d = os.path.join(OUTPUTS, "{}_lora{}".format(stage, tag))
    if not os.path.isdir(d):
        return None
    cands = []
    root_ts = os.path.join(d, "trainer_state.json")
    if os.path.exists(root_ts):
        cands.append(root_ts)
    for name in os.listdir(d):
        p = os.path.join(d, name, "trainer_state.json")
        if name.startswith("checkpoint-") and os.path.exists(p):
            cands.append(p)
    if not cands:
        return None
    # 选 log_history 最长(=最新/最全)的
    return max(cands, key=lambda p: len(json.load(open(p)).get("log_history", [])))


def load_history(stage, tag):
    """返回 (source, rows) — rows 为 dict 列表(键含 loss/mean_token_accuracy/rewards|epoch...)。"""
    ts = find_trainer_state(stage, tag)
    if ts:
        hist = json.load(open(ts, encoding="utf-8")).get("log_history", [])
        return "trainer_state.json", hist
    # 回退解析日志
    path = LOGS[stage]
    if not os.path.exists(path):
        return None, []
    txt = open(path, encoding="utf-8", errors="ignore").read()
    rows = []
    for m in RE_BLOCK.finditer(txt):
        b = m.group(0)
        rows.append({"loss": _kv(b, "loss"), "grad_norm": _kv(b, "grad_norm"),
                     "epoch": _kv(b, "epoch"), "mean_token_accuracy": _kv(b, "mean_token_accuracy"),
                     "rewards/accuracies": _kv(b, "rewards/accuracies"),
                     "rewards/margins": _kv(b, "rewards/margins")})
    return "chain_{}.log".format(stage), rows


def series(rows, key):
    xs, ys = [], []
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is not None:
            xs.append(r.get("step", r.get("epoch", i)) if key != "eval_loss" else r.get("step", i))
            ys.append(v)
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["auto", "sft", "dpo"], default="auto")
    ap.add_argument("--tag", default="_v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    stage = a.stage
    if stage == "auto":
        # 优先有 trainer_state 的；否则按日志 mtime
        for s in ("dpo", "sft"):
            if find_trainer_state(s, a.tag):
                stage = s; break
        if stage == "auto":
            existing = [(s, LOGS[s]) for s in ("sft", "dpo") if os.path.exists(LOGS[s])]
            stage = max(existing, key=lambda sp: os.path.getmtime(sp[1]))[0] if existing else "sft"

    src, rows = load_history(stage, a.tag)
    if not rows:
        print("无可用数据(trainer_state / 日志均无 loss 记录)，stage=%s" % stage)
        return
    out_png = a.out or os.path.join(LOGDIR, "{}_curve{}.png".format(stage, a.tag))
    out_csv = a.csv or os.path.join(LOGDIR, "{}_curve{}.csv".format(stage, a.tag))

    # CSV：合并所有出现过的键
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # PNG
    lx = [i for i, r in enumerate(rows) if r.get("loss") is not None]
    ly = [r["loss"] for r in rows if r.get("loss") is not None]
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.plot(lx, ly, color="#378ADD", label="train loss")
    ax1.set_xlabel("logging step"); ax1.set_ylabel("loss", color="#378ADD")
    # eval_loss 散点(若有)
    ex = [i for i, r in enumerate(rows) if r.get("eval_loss") is not None]
    ey = [r["eval_loss"] for r in rows if r.get("eval_loss") is not None]
    if ey:
        ax1.scatter(ex, ey, color="#C0392B", marker="o", s=30, label="eval_loss", zorder=5)
    ax1.legend(loc="upper right")

    sec_key = "rewards/accuracies" if stage == "dpo" else "mean_token_accuracy"
    sy = [r.get(sec_key) for r in rows if r.get(sec_key) is not None]
    if sy:
        sx = [i for i, r in enumerate(rows) if r.get(sec_key) is not None]
        ax2 = ax1.twinx()
        ax2.plot(sx, sy, color="#E07B39", alpha=0.75)
        ax2.set_ylabel(sec_key, color="#E07B39")
        if stage == "dpo":
            my = [r.get("rewards/margins") for r in rows if r.get("rewards/margins") is not None]
            if my:
                mx = [i for i, r in enumerate(rows) if r.get("rewards/margins") is not None]
                ax2.plot(mx, my, color="#27AE60", alpha=0.55, linestyle="--")
    ax1.set_title("{} training curve (Qwen3.5-2B QLoRA, tag={}) · src={}".format(stage.upper(), a.tag, src))
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print("stage=%s source=%s rows=%d" % (stage, src, len(rows)))
    print("written:", out_png)
    print("written:", out_csv)


if __name__ == "__main__":
    main()
