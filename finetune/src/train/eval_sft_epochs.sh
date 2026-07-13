#!/usr/bin/env bash
# eval_sft_epochs.sh — 等当前 SFT 任务跑完后，补评倒数第二个 checkpoint(ep2)并出逐轮对比。
#
# 背景：训练用 --no-eval(用满数据)，save_strategy=epoch + save_total_limit=2 会保留
#       最后两轮 checkpoint(ep2、ep3)。主链路已评 ep3(=根适配器, slug=...sft${TAG})；
#       本脚本补评 ep2，再用 compare_epochs.py 以 fc_eval 真实指标选优/判过拟合。
#
# 用法:
#   PY=<venv-python> [WAIT_PID=<训练pid>] bash src/train/eval_sft_epochs.sh <tag>
set -euo pipefail

TAG="${1:-_v1}"
PY="${PY:-python}"
WAIT_PID="${WAIT_PID:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # finetune/src/train
ROOT="$(cd "$HERE/../.." && pwd)"                        # finetune/
FC="$(cd "$ROOT/../fc_eval" && pwd)"
EVAL_HF="$ROOT/src/eval/eval_hf.py"
CMP="$ROOT/src/eval/compare_epochs.py"
LOGDIR="$ROOT/logs/pipeline"; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/chain_status.txt"
SFT_OUT="$ROOT/outputs/sft_lora${TAG}"
EP3_SLUG="qwen3.5-2b-hf-sft${TAG}"       # 主链路已评(=ep3/根)
EP2_SLUG="qwen3.5-2b-hf-sft${TAG}_ep2"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ep-cmp] $*" | tee -a "$STATUS"; }

# 1) 等主训练任务结束(避免 GPU 争用)
if [ -n "$WAIT_PID" ]; then
  log "等待训练进程 pid=$WAIT_PID 结束 ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "pid=$WAIT_PID 已结束"
fi
sleep 5

# 2) 校验 SFT 产物与 ep3 评测已就绪
if [ ! -f "$SFT_OUT/adapter_model.safetensors" ]; then
  log "SFT 产物缺失($SFT_OUT)，可能训练失败，终止 ep 对比"; exit 1
fi
if [ ! -f "$FC/results/$EP3_SLUG/metrics.json" ]; then
  log "ep3 评测 metrics 缺失，补跑 ep3 评测 ..."
  "$PY" -u "$EVAL_HF" --slug "$EP3_SLUG" --adapter "outputs/sft_lora${TAG}" 2>&1 | tee "$LOGDIR/chain_eval_sft.log"
  ( cd "$FC" && "$PY" -u scorer.py --model "$EP3_SLUG" ) 2>&1 | tee "$LOGDIR/chain_score_sft.log"
fi

# 3) 找 ep2 checkpoint(保留的最小步号 checkpoint) 并评测
EP2_CKPT="$(ls -d "$SFT_OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -1 || true)"
MAX_CKPT="$(ls -d "$SFT_OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
if [ -z "$EP2_CKPT" ] || [ "$EP2_CKPT" = "$MAX_CKPT" ]; then
  log "未找到独立的 ep2 checkpoint(仅一个或无)，跳过 ep2 评测，仅对比 base vs ep3。"
else
  REL_EP2="${EP2_CKPT#$ROOT/}"   # 相对 finetune/ 的路径(eval_hf 用 ROOT 拼接)
  log "评测 ep2 checkpoint: $EP2_CKPT (slug=$EP2_SLUG)"
  "$PY" -u "$EVAL_HF" --slug "$EP2_SLUG" --adapter "$REL_EP2" 2>&1 | tee "$LOGDIR/chain_eval_sft_ep2.log"
  ( cd "$FC" && "$PY" -u scorer.py --model "$EP2_SLUG" ) 2>&1 | tee "$LOGDIR/chain_score_sft_ep2.log"
fi

# 4) 逐轮对比 + 选优
log "生成逐轮对比 compare_sft_epochs.md ..."
if [ -f "$FC/results/$EP2_SLUG/metrics.json" ]; then
  "$PY" -u "$CMP" "base:qwen3.5-2b-hf-base" "SFT-ep2:$EP2_SLUG" "SFT-ep3:$EP3_SLUG" 2>&1 | tee "$LOGDIR/chain_compare_epochs.log"
else
  "$PY" -u "$CMP" "base:qwen3.5-2b-hf-base" "SFT-ep3:$EP3_SLUG" 2>&1 | tee "$LOGDIR/chain_compare_epochs.log"
fi
log "EP 对比完成 -> fc_eval/results/compare_sft_epochs.md"
