#!/usr/bin/env bash
# run_sft_eval.sh — 仅 SFT -> SFT 评测(不含 DPO)。
#
# 用途:当预估 SFT+DPO+评测总耗时 > 8h 时(见 MEMORY「训练编排耗时阈值规则」),
# 完成 SFT 后先做 SFT 评测,DPO 暂缓。SFT adapter 保留,后续可单独承接 DPO。
#
# 用法:
#   PY=<venv-python> bash src/train/run_sft_eval.sh <tag> <sft_ep>
#   例: PY=/apdcephfs/.../finetune_env312/bin/python bash src/train/run_sft_eval.sh _v1 3
set -euo pipefail

TAG="${1:-_v1}"
SFT_EPOCHS="${2:-3}"
PY="${PY:-python}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # finetune/src/train
ROOT="$(cd "$HERE/../.." && pwd)"                        # finetune/
FC="$(cd "$ROOT/../fc_eval" && pwd)"                     # fc_eval/
TRAIN_SFT="$HERE/train_sft.py"
EVAL_HF="$ROOT/src/eval/eval_hf.py"
LOGDIR="$ROOT/logs/pipeline"
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/chain_status.txt"
SFT_OUT="$ROOT/outputs/sft_lora${TAG}"
SFT_SLUG="qwen3.5-2b-hf-sft${TAG}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }

: > "$STATUS"
log "SFT-ONLY CHAIN START tag=${TAG} sft_ep=${SFT_EPOCHS} PY=${PY} (DPO 暂缓, 依据>8h规则)"

log "PREFLIGHT: 清理残留 python 进程"
pkill -u "$(id -un)" -f "train_sft.py|train_dpo.py|eval_hf.py" 2>/dev/null || true
sleep 3

# ============ 1) SFT ============
log "SFT START -> $SFT_OUT"
"$PY" -u "$TRAIN_SFT" --tag "$TAG" --epochs "$SFT_EPOCHS" --no-eval 2>&1 | tee "$LOGDIR/chain_sft.log"

if [ ! -f "$SFT_OUT/adapter_model.safetensors" ]; then
  log "SFT FAILED: 未找到 $SFT_OUT/adapter_model.safetensors,终止"
  exit 1
fi
log "SFT DONE (产物校验通过)"

# ============ 2) SFT 评测 (不做 DPO) ============
log "EVAL SFT START -> $SFT_SLUG"
"$PY" -u "$EVAL_HF" --slug "$SFT_SLUG" --adapter "outputs/sft_lora${TAG}" 2>&1 | tee "$LOGDIR/chain_eval_sft.log"
( cd "$FC" && "$PY" -u scorer.py --model "$SFT_SLUG" ) 2>&1 | tee "$LOGDIR/chain_score_sft.log"
log "EVAL SFT DONE: results/${SFT_SLUG}"

log "SFT-ONLY CHAIN DONE. DPO 暂缓(如需承接: PY=<venv> $PY -u $HERE/train_dpo.py --tag ${TAG} --epochs 2 --beta 0.1)"
