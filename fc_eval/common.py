"""common.py — 多模型评测的共用路径工具。

模型名含冒号(qwen3.5:2b),Windows 文件名/路径非法,统一 slug 化为 qwen3.5-2b。
每个模型的结果隔离到 results/<slug>/ 子目录,避免互相覆盖。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def model_slug(model):
    """qwen3.5:2b -> qwen3.5-2b ; qwen3.5:9b -> qwen3.5-9b（保留点号）。"""
    return model.replace(":", "-").replace("/", "-").replace("\\", "-")


def results_dir(model, ensure=False, suffix=""):
    """model 的结果目录;suffix 非空时追加(如 '-ablation')隔离不同实验,保护基线。"""
    name = model_slug(model) + (suffix or "")
    d = os.path.join(HERE, "results", name)
    if ensure:
        os.makedirs(d, exist_ok=True)
    return d
