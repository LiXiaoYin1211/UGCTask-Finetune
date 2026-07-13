#!/usr/bin/env bash
# setup_env.sh — 在工作区内搭建 Python 3.12 训练环境（H20 / CUDA 12.1）。
# 依赖按平台自动选最新兼容版（不强锁 pin）；torch 用 cu121。
# 所有缓存/解释器/venv 均落在工作区内。日志: finetune/logs/env_setup.log
set -uo pipefail

WS=/apdcephfs/private_hynnzhang/ServerFiles
export UV_PYTHON_INSTALL_DIR=$WS/.uv_python
export UV_CACHE_DIR=$WS/.uv_cache
export PIP_CACHE_DIR=$WS/.pip_cache
PY312=$WS/.uv_python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
VENV=$WS/finetune_env312
PIP="$VENV/bin/pip"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# --- 单实例锁：防止重复启动并发写坏 venv ---
LOCK=$WS/finetune/logs/.setup_env.lock
exec 9>"$LOCK" || { echo "cannot open lock"; exit 1; }
if ! flock -n 9; then
  echo "[$(ts)] ANOTHER setup_env.sh IS RUNNING — abort this instance."
  exit 0
fi

echo "[$(ts)] ===== SETUP START ====="

# 1) venv：干净重建（上次并发安装可能污染，直接重来确保干净）
echo "[$(ts)] (re)create clean venv (py3.12) -> $VENV"
rm -rf "$VENV"
"$PY312" -m venv "$VENV" || { echo "[$(ts)] FATAL venv create failed"; exit 1; }
echo "[$(ts)] python: $($VENV/bin/python --version 2>&1)"

# 2) 基础工具
echo "[$(ts)] upgrade pip/setuptools/wheel"
"$PIP" install --upgrade pip setuptools wheel 2>&1 | tail -2

# 3) torch (cu121, 适配 H20 sm_90 / CUDA12.1)
echo "[$(ts)] install torch (cu121) ..."
"$PIP" install torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5
echo "[$(ts)] torch install rc=$?"

# 4) 训练/数据依赖（最新兼容版，不锁 pin）
echo "[$(ts)] install transformers/peft/trl/bitsandbytes/datasets/accelerate/... "
"$PIP" install transformers peft trl bitsandbytes datasets accelerate safetensors tokenizers jsonschema matplotlib numpy 2>&1 | tail -8
echo "[$(ts)] deps install rc=$?"

# 5) 冻结版本快照（留档工作区）
echo "[$(ts)] freeze -> $WS/finetune/logs/env_freeze.txt"
"$PIP" freeze > "$WS/finetune/logs/env_freeze.txt" 2>&1

# 6) 关键校验：torch+GPU、核心库版本、bitsandbytes
echo "[$(ts)] ===== VERIFY ====="
"$VENV/bin/python" - <<'PYEOF' 2>&1
import importlib
def v(m):
    try:
        mod = importlib.import_module(m)
        return getattr(mod, "__version__", "?")
    except Exception as e:
        return f"IMPORT-FAIL: {type(e).__name__}: {e}"
import torch
print("torch     :", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device    :", torch.cuda.get_device_name(0), "| sm_", "".join(map(str, torch.cuda.get_device_capability(0))))
    a=torch.randn(512,512,device="cuda",dtype=torch.bfloat16); b=torch.randn(512,512,device="cuda",dtype=torch.bfloat16)
    print("bf16 matmul OK, sum=", round((a@b).float().sum().item(),2))
for m in ["transformers","peft","trl","bitsandbytes","datasets","accelerate","safetensors","tokenizers","numpy"]:
    print(f"{m:12s}:", v(m))
PYEOF
echo "[$(ts)] ===== SETUP DONE ====="
