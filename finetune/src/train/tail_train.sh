#!/usr/bin/env bash
# tail_train.sh — 通用训练日志跟随(SFT/DPO 自动切换)。
#
# 用 `tail -F`(按文件名跟随 + 文件出现/轮转自动重试)同时盯 chain_sft.log 与 chain_dpo.log：
# - SFT 阶段只有 chain_sft.log 增长 -> 显示 SFT;
# - DPO 阶段 chain_dpo.log 开始增长 -> 自动接上显示 DPO(带 ==> 文件 <== 头);
# - 目标文件暂不存在也不报错退出,出现后自动开始跟随。
#
# 用法:
#   bash src/train/tail_train.sh          # 默认各显示末 20 行后持续跟随
#   bash src/train/tail_train.sh 50       # 末 50 行
# 提示: 原始日志含 tqdm 的 \r 进度条,较"乱";要清爽面板用 monitor_train.py --watch。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # finetune/src/train
LOGDIR="$(cd "$HERE/../.." && pwd)/logs/pipeline"      # finetune/logs/pipeline
N="${1:-20}"

exec tail -n "$N" -F "$LOGDIR/chain_sft.log" "$LOGDIR/chain_dpo.log"
