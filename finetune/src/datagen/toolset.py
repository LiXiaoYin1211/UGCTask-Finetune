"""toolset.py — 训练数据的工具表单一真源。

大显存服务器方案:每条训练样本携带全 22 完整工具(含 description),
与推理端 eval_hf/runner 用的 fc_eval/tools_ugc.json 完全同源 -> 训推一致。
gen_sft / gen_dpo 直接引用 FULL_TOOLS 写入每条样本的 tools 字段。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/
TOOLS_PATH = os.path.join(ROOT, "..", "fc_eval", "tools_ugc.json")

# 全 22 完整工具(含 description),与推理端 eval_hf/runner 的 tools_ugc.json 同源。
FULL_TOOLS = json.load(open(TOOLS_PATH, encoding="utf-8"))["tools"]
ALL_NAMES = [t["function"]["name"] for t in FULL_TOOLS]
