#!/usr/bin/env bash
# run_train_chain.sh — SFT -> DPO 自包含串联训练脚本(Linux 服务器版)。
#
# 设计要点:
# - set -euo pipefail:任一步失败立即整体退出,不带坏状态往下跑。
# - SFT/DPO 用显式产物校验(adapter_model.safetensors 是否存在)串联,
#   SFT 真正产出权重才进入 DPO,规避"进程退了但没存 checkpoint"的假成功。
# - SFT->DPO 是"训练->训练"衔接,依赖固化在脚本 && 里,与调用方是否存活无关。
# - 评测默认单独跑;传第 5 个参数 with_eval=1 可在训练后自动接评测(大显存服务器无时序坑)。
#
# 用法:
#   PY=python bash src/train/run_train_chain.sh <tag> <sft_ep> <dpo_ep> <beta> [with_eval]
#   例: PY=/opt/venv/bin/python bash src/train/run_train_chain.sh _v4 2 1 0.3 1
# 环境变量:
#   PY          python 可执行文件(默认 "python");建议指向目标 venv 的 python。
#   STOP_OLLAMA 设为 1 时尝试停 Ollama 释放显存(默认不动)。
set -euo pipefail

TAG="${1:-_v4}"
SFT_EPOCHS="${2:-2}"
DPO_EPOCHS="${3:-1}"
DPO_BETA="${4:-0.3}"
WITH_EVAL="${5:-0}"

PY="${PY:-python}"

# --- 路径(脚本自定位,与 cwd 无关;纯 POSIX 路径) ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # finetune/src/train
ROOT="$(cd "$HERE/../.." && pwd)"                        # finetune/
FC="$(cd "$ROOT/../fc_eval" && pwd)"                     # fc_eval/
TRAIN_SFT="$HERE/train_sft.py"
TRAIN_DPO="$HERE/train_dpo.py"
EVAL_HF="$ROOT/src/eval/eval_hf.py"
LOGDIR="$ROOT/logs/pipeline"
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/chain_status.txt"
SFT_OUT="$ROOT/outputs/sft_lora${TAG}"
DPO_OUT="$ROOT/outputs/dpo_lora${TAG}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }

: > "$STATUS"
log "CHAIN START tag=${TAG} sft_ep=${SFT_EPOCHS} dpo_ep=${DPO_EPOCHS} beta=${DPO_BETA} with_eval=${WITH_EVAL} PY=${PY}"

# --- Preflight:清理本用户残留 python 训练进程,可选停 Ollama(释放显存) ---
log "PREFLIGHT: 清理残留 python 进程"
pkill -u "$(id -un)" -f "train_sft.py|train_dpo.py|eval_hf.py" 2>/dev/null || true
if [ "${STOP_OLLAMA:-0}" = "1" ]; then
  log "PREFLIGHT: 停止 Ollama"
  (systemctl stop ollama 2>/dev/null || pkill -f "ollama" 2>/dev/null) || true
fi
sleep 3

# ============ 1) SFT ============
log "SFT START -> $SFT_OUT"
"$PY" -u "$TRAIN_SFT" --tag "$TAG" --epochs "$SFT_EPOCHS" --no-eval 2>&1 | tee "$LOGDIR/chain_sft.log"

# 显式校验 SFT 产物(避免假成功)
if [ ! -f "$SFT_OUT/adapter_model.safetensors" ]; then
  log "SFT FAILED: 未找到 $SFT_OUT/adapter_model.safetensors,终止,不进入 DPO"
  exit 1
fi
log "SFT DONE (产物校验通过)"
# 训练曲线持久化(loss/acc PNG + CSV，来源 trainer_state.json)。失败不阻断主流程。
"$PY" -u "$ROOT/src/eval/plot_curves.py" --stage sft --tag "$TAG" 2>&1 | tee "$LOGDIR/chain_plot_sft.log" || true

# ============ 2) DPO(承接 SFT adapter) ============
# train_dpo.py 默认用 sft_lora${TAG} 作为 SFT adapter,无需显式传 --sft
log "DPO START -> $DPO_OUT (承接 $SFT_OUT)"
"$PY" -u "$TRAIN_DPO" --tag "$TAG" --epochs "$DPO_EPOCHS" --beta "$DPO_BETA" 2>&1 | tee "$LOGDIR/chain_dpo.log"

if [ ! -f "$DPO_OUT/adapter_model.safetensors" ]; then
  log "DPO FAILED: 未找到 $DPO_OUT/adapter_model.safetensors"
  exit 1
fi
log "DPO DONE (产物校验通过)"
# 训练曲线持久化(loss/rewards_acc/margin PNG + CSV)。失败不阻断主流程。
"$PY" -u "$ROOT/src/eval/plot_curves.py" --stage dpo --tag "$TAG" 2>&1 | tee "$LOGDIR/chain_plot_dpo.log" || true
log "CHAIN(train) DONE: 产物 -> $SFT_OUT, $DPO_OUT"

# ============ 3) 可选:评测(with_eval=1) ============
if [ "$WITH_EVAL" = "1" ]; then
  SFT_SLUG="qwen3.5-2b-hf-sft${TAG}"
  DPO_SLUG="qwen3.5-2b-hf-dpo${TAG}"
  log "EVAL SFT START -> $SFT_SLUG"
  "$PY" -u "$EVAL_HF" --slug "$SFT_SLUG" --adapter "outputs/sft_lora${TAG}" 2>&1 | tee "$LOGDIR/chain_eval_sft.log"
  ( cd "$FC" && "$PY" -u scorer.py --model "$SFT_SLUG" ) 2>&1 | tee "$LOGDIR/chain_score_sft.log"
  log "EVAL DPO START -> $DPO_SLUG"
  "$PY" -u "$EVAL_HF" --slug "$DPO_SLUG" --adapter "outputs/dpo_lora${TAG}" 2>&1 | tee "$LOGDIR/chain_eval_dpo.log"
  ( cd "$FC" && "$PY" -u scorer.py --model "$DPO_SLUG" ) 2>&1 | tee "$LOGDIR/chain_score_dpo.log"
  log "EVAL DONE: results/${SFT_SLUG}, results/${DPO_SLUG}"
fi

log "CHAIN DONE."
