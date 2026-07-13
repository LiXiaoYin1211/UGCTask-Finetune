#!/usr/bin/env bash
# eval_ep3_ep4.sh — 等 overall-ep4 续训(outputs/sft_lora_v1_ep4)结束后，自动评测
# overall ep3(checkpoint-63) 与 overall ep4(根适配器)，并出五方对比 base/ep1/ep2/ep3/ep4。
#
# 背景：续训以 ep2 为 init、fresh 训 2 epoch → 本轮内部 ep1/ep2 = overall ep3/ep4。
#       评测沿用 max_new=200，与 base/ep1/ep2 同配置保证可比。
#
# 用法:
#   PY=<venv-python> [WAIT_PID=<续训pid>] bash src/train/eval_ep3_ep4.sh
set -uo pipefail

PY="${PY:-python}"
WAIT_PID="${WAIT_PID:-}"
TAG_EP4="_v1_ep4"
MAXNEW="${MAXNEW:-200}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # finetune/src/train
ROOT="$(cd "$HERE/../.." && pwd)"                        # finetune/
FC="$(cd "$ROOT/../fc_eval" && pwd)"
EVAL_HF="$ROOT/src/eval/eval_hf.py"
CMP="$ROOT/src/eval/compare_epochs.py"
LOGDIR="$ROOT/logs/pipeline"; mkdir -p "$LOGDIR"
STATUS="$LOGDIR/chain_status.txt"
OUT4="$ROOT/outputs/sft_lora${TAG_EP4}"          # outputs/sft_lora_v1_ep4
EP3_CKPT="$OUT4/checkpoint-63"                    # overall ep3
EP3_SLUG="qwen3.5-2b-hf-sft_v1_ep3"
EP4_SLUG="qwen3.5-2b-hf-sft_v1_ep4"

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ep34] $*" | tee -a "$STATUS"; }

# 1) 等续训结束(避免 GPU 争用)
if [ -n "$WAIT_PID" ]; then
  log "等待续训 pid=$WAIT_PID 结束 ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "pid=$WAIT_PID 已结束"
fi
# 再等根产物落盘(save_model 完成)，最多 ~10 分钟
for i in $(seq 1 40); do
  [ -f "$OUT4/adapter_model.safetensors" ] && break
  sleep 15
done
if [ ! -f "$OUT4/adapter_model.safetensors" ]; then
  log "未找到 $OUT4/adapter_model.safetensors，续训可能失败，终止"; exit 1
fi
sleep 5

# 2) 评 overall ep4 (根适配器)
log "EVAL ep4(根) -> $EP4_SLUG (max_new=$MAXNEW)"
"$PY" -u "$EVAL_HF" --slug "$EP4_SLUG" --adapter "outputs/sft_lora${TAG_EP4}" --max-new "$MAXNEW" 2>&1 | tee "$LOGDIR/chain_eval_sft_ep4.log"
( cd "$FC" && "$PY" -u scorer.py --model "$EP4_SLUG" ) 2>&1 | tee "$LOGDIR/chain_score_sft_ep4.log"

# 3) 评 overall ep3 (checkpoint-63)
if [ -d "$EP3_CKPT" ]; then
  log "EVAL ep3(checkpoint-63) -> $EP3_SLUG (max_new=$MAXNEW)"
  "$PY" -u "$EVAL_HF" --slug "$EP3_SLUG" --adapter "outputs/sft_lora${TAG_EP4}/checkpoint-63" --max-new "$MAXNEW" 2>&1 | tee "$LOGDIR/chain_eval_sft_ep3.log"
  ( cd "$FC" && "$PY" -u scorer.py --model "$EP3_SLUG" ) 2>&1 | tee "$LOGDIR/chain_score_sft_ep3.log"
else
  log "未找到 $EP3_CKPT，跳过 ep3 评测"
fi

# 4) 五方对比 base/ep1/ep2/ep3/ep4
log "生成五方对比 base/ep1/ep2/ep3/ep4 ..."
"$PY" -u "$CMP" \
  "base:qwen3.5-2b-hf-base" \
  "SFT-ep1:qwen3.5-2b-hf-sft_v1_ep1" \
  "SFT-ep2:qwen3.5-2b-hf-sft_v1" \
  "SFT-ep3:$EP3_SLUG" \
  "SFT-ep4:$EP4_SLUG" 2>&1 | tee "$LOGDIR/chain_compare_5way.log"
log "五方对比完成 -> fc_eval/results/compare_sft_epochs.md"
