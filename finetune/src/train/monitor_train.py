"""monitor_train.py — 训练过程/进度可视化(终端 + 可选 PNG)，SFT / DPO 通用。

非侵入式：仅解析 logs/pipeline/ 下的日志，不改动、不干扰正在运行的训练进程。
- 自动识别当前阶段：在 chain_sft.log / chain_dpo.log 中选『最近更新』的那个（可用 --stage 强制）。
- SFT 展示 loss/grad_norm/mean_token_accuracy；DPO 展示 loss/grad_norm/rewards_acc/margin。
展示：阶段、进度条、已用/剩余(ETA)、s/it、最新指标、loss 趋势(ASCII 火花线)、GPU。

用法:
  python monitor_train.py                 # 打印一次(自动选 SFT/DPO 日志)
  python monitor_train.py --watch 10      # 每 10s 刷新
  python monitor_train.py --stage dpo     # 强制看 DPO 日志
  python monitor_train.py --log chain_sft_cont_ep4.log --watch 10  # 指定任意日志文件(相对 logs/pipeline/ 或绝对路径)
  python monitor_train.py --png curve.png # 另存 loss 曲线 PNG(快照)
"""
import argparse
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/
LOGDIR = os.path.join(ROOT, "logs", "pipeline")
LOGS = {"sft": os.path.join(LOGDIR, "chain_sft.log"),
        "dpo": os.path.join(LOGDIR, "chain_dpo.log")}
STATUS = os.path.join(LOGDIR, "chain_status.txt")
EVAL_PROG = os.path.join(LOGDIR, "eval_hf_progress.txt")

SPARK = "▁▂▃▄▅▆▇█"
RE_BLOCK = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
RE_TQDM = re.compile(r"(\d+)/(\d+)\s*\[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?),\s*([\d.]+)s/it")
RE_EVAL = re.compile(r"\[(\d+)/(\d+)\]")


def _kv(block, key):
    m = re.search(r"'%s':\s*'?(-?[\d.eE+]+)'?" % re.escape(key), block)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def tail_text(path, nbytes=400000):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", "ignore")


def pick_log(stage):
    """返回 (stage_name, path)。auto: 选最近更新且存在的日志。"""
    if stage in ("sft", "dpo") and os.path.exists(LOGS[stage]):
        return stage, LOGS[stage]
    cands = [(s, p) for s, p in LOGS.items() if os.path.exists(p)]
    if not cands:
        return "sft", LOGS["sft"]
    cands.sort(key=lambda sp: os.path.getmtime(sp[1]), reverse=True)
    return cands[0]


def resolve_log(stage, log_override=None):
    """返回 (stage_name, path)。若给了 log_override(相对 LOGDIR 或绝对路径)则优先用它，
    stage 名用于选择副指标(SFT: mean_token_accuracy / DPO: rewards)。"""
    if log_override:
        path = log_override if os.path.isabs(log_override) else os.path.join(LOGDIR, log_override)
        if stage in ("sft", "dpo"):
            st = stage
        else:
            st = "dpo" if "dpo" in os.path.basename(path).lower() else "sft"
        return st, path
    return pick_log(stage)


def parse_records(txt):
    recs = []
    for m in RE_BLOCK.finditer(txt):
        b = m.group(0)
        recs.append({
            "loss": _kv(b, "loss"), "gn": _kv(b, "grad_norm"), "epoch": _kv(b, "epoch"),
            "acc": _kv(b, "mean_token_accuracy"),
            "racc": _kv(b, "rewards/accuracies"), "margin": _kv(b, "rewards/margins"),
        })
    return recs


def sparkline(vals, width=48):
    v = [x for x in vals if x is not None][-width:]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((x - lo) / rng * (len(SPARK) - 1)))] for x in v)


def bar(cur, tot, width=40):
    frac = 0 if not tot else cur / tot
    fill = int(frac * width)
    return "[" + "█" * fill + "·" * (width - fill) + "] {:.1f}%".format(frac * 100)


def gpu_line():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
        u, t, g, temp = [x.strip() for x in out.split(",")]
        return "GPU {}/{} MiB ({:.0f}%显存) · util {}% · {}°C".format(u, t, 100 * float(u) / float(t), g, temp)
    except Exception:
        return "GPU: (nvidia-smi 不可用)"


def status_last():
    lines = [l for l in tail_text(STATUS, 8000).splitlines() if l.strip()]
    return lines[-1] if lines else "(无状态)"


def render(stage_arg, log_override=None):
    stage, log = resolve_log(stage_arg, log_override)
    txt = tail_text(log)
    recs = parse_records(txt)
    tq = RE_TQDM.findall(txt)
    L = []
    L.append("=" * 70)
    L.append("训练监控 [{}]  {}".format(stage.upper(), time.strftime("%Y-%m-%d %H:%M:%S")))
    L.append("状态: " + status_last())
    L.append("日志: " + os.path.basename(log) + ("" if os.path.exists(log) else "  (文件不存在!)"))
    L.append("-" * 70)
    if tq:
        cur, tot, el, eta, sit = tq[-1]
        L.append("{} 进度  {}".format(stage.upper(), bar(int(cur), int(tot))))
        L.append("  step {}/{} · 已用 {} · 剩余(ETA) {} · {}s/it".format(cur, tot, el, eta, sit))
    else:
        L.append("{} 进度: (尚无 tqdm 行；可能在预处理/加载)".format(stage.upper()))
    if recs:
        r = recs[-1]
        base = "  最新: loss {:.4f} · grad_norm {:.3f} · epoch {:.2f}".format(
            r["loss"] or 0, r["gn"] or 0, r["epoch"] or 0)
        if stage == "dpo":
            extra = ""
            if r["racc"] is not None:
                extra += " · reward_acc {:.3f}".format(r["racc"])
            if r["margin"] is not None:
                extra += " · margin {:.3f}".format(r["margin"])
            L.append(base + extra)
            L.append("  loss 趋势 " + sparkline([x["loss"] for x in recs]))
            if any(x["racc"] is not None for x in recs):
                L.append("  r_acc趋势 " + sparkline([x["racc"] for x in recs]))
        else:
            if r["acc"] is not None:
                base += " · acc {:.4f}".format(r["acc"])
            L.append(base)
            L.append("  loss 趋势 " + sparkline([x["loss"] for x in recs]))
            if any(x["acc"] is not None for x in recs):
                L.append("  acc  趋势 " + sparkline([x["acc"] for x in recs]))
    # 评测进度(仅当评测日志近 5 分钟内更新过)
    fresh = os.path.exists(EVAL_PROG) and (time.time() - os.path.getmtime(EVAL_PROG) < 300)
    evm = RE_EVAL.findall(tail_text(EVAL_PROG, 20000))
    if evm and fresh:
        L.append("-" * 70)
        L.append("评测进度: {}/{} 例".format(evm[-1][0], evm[-1][1]))
    L.append("-" * 70)
    L.append(gpu_line())
    L.append("=" * 70)
    return "\n".join(L)


def save_png(path, stage_arg, log_override=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stage, log = resolve_log(stage_arg, log_override)
    recs = parse_records(tail_text(log, 4_000_000))
    if not recs:
        print("no data yet in", os.path.basename(log))
        return
    ys = [x["loss"] for x in recs]
    xs = list(range(1, len(ys) + 1))
    sec_key, sec_lbl = ("racc", "rewards/accuracies") if stage == "dpo" else ("acc", "mean_token_accuracy")
    sec = [x[sec_key] for x in recs]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(xs, ys, color="#378ADD", label="loss")
    ax1.set_xlabel("logging step"); ax1.set_ylabel("loss", color="#378ADD")
    if any(v is not None for v in sec):
        ax2 = ax1.twinx()
        ax2.plot(xs, [v if v is not None else float("nan") for v in sec], color="#E07B39", alpha=0.7)
        ax2.set_ylabel(sec_lbl, color="#E07B39")
    ax1.set_title("{} training curve (Qwen3.5-2B QLoRA)".format(stage.upper()))
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print("written:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="刷新间隔秒；0=打印一次")
    ap.add_argument("--stage", choices=["auto", "sft", "dpo"], default="auto", help="强制阶段；默认自动")
    ap.add_argument("--log", default=None,
                    help="指定要解析的训练日志文件(相对 logs/pipeline/ 或绝对路径)；覆盖自动选择。"
                         "用于日志名非 chain_sft.log/chain_dpo.log 的情形(如续训 chain_sft_cont_ep4.log)。")
    ap.add_argument("--png", default=None, help="另存 loss 曲线 PNG 路径")
    a = ap.parse_args()
    if a.png:
        save_png(a.png, a.stage, a.log)
        return
    if a.watch <= 0:
        print(render(a.stage, a.log))
        return
    try:
        while True:
            os.system("clear")
            print(render(a.stage, a.log))
            print("\n(每 {}s 刷新，Ctrl-C 退出)".format(a.watch))
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
