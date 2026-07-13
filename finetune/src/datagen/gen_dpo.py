"""gen_dpo.py — 按覆盖矩阵生成 DPO 偏好对,输出 data/dpo.jsonl。

覆盖矩阵(规格):
  该问不问→ask_user 22% · ID幻觉 20% · 错选工具(place↔create) 18% ·
  跳过前置依赖 15% · 过度触发ask_user(反向) 13% · 格式/必填 7% · 越界/幻觉工具名 5%
chosen/rejected 仅在关键决策点分叉; rejected 的 tool 结果用真实错误串。
禁 think。与评测集去重交给 validate.py。

用法: python gen_dpo.py --n 800 [--seed 0]
"""
import argparse
import json
import os
import random
import re

import scene_db
import prototypes as P
import toolset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
EVAL_PATH = os.path.join(ROOT, "..", "fc_eval", "dataset.jsonl")


def _tool_names_in(msgs):
    """抽取一段 messages 里出现的所有 assistant 工具调用名(用于构造子集,确保正确/错误工具都在候选内)。"""
    names = []
    for m in msgs or []:
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("function", {}).get("name")
            if nm:
                names.append(nm)
    return names


def _norm(s):
    return re.sub(r"\s+", "", (s or "").lower())


def _eval_queries():
    qs = set()
    if os.path.exists(EVAL_PATH):
        for line in open(EVAL_PATH, encoding="utf-8"):
            line = line.strip()
            if line:
                qs.add(_norm(json.loads(line)["query"]))
    return qs


EVAL_Q = _eval_queries()

MATRIX = [
    (P.dpo_should_ask, 0.22),
    (P.dpo_id_hallucination, 0.20),
    (P.dpo_place_vs_create, 0.18),
    (P.dpo_skip_dependency, 0.15),
    (P.dpo_over_ask, 0.13),
    (P.dpo_format_required, 0.07),
    (P.dpo_oob_tool, 0.05),
]

# query 近义扩展(给每类几个说法,增多样、避免与评测集撞)
PARAPHRASE = {
    "dpo_should_ask": ["把那个弄好看点", "随便美化一下", "帮我整得高级点", "弄漂亮些"],
    "dpo_id_hallucination": ["把 id {n} 的桌子删了", "删掉 id {n} 这种桌子", "id {n} 的桌子不要了"],
    "dpo_place_vs_create": ["在桌子上放个台灯", "桌上摆盏灯", "给桌子上放个灯"],
    "dpo_skip_dependency": ["给那面墙刷成蓝色", "把墙刷蓝", "墙面涂成蓝色"],
    "dpo_over_ask": ["我附近有什么椅子", "附近有哪些椅子", "我周围的椅子"],
    "dpo_format_required": ["目录里有哪些花瓶", "找几个花瓶看看", "有什么花瓶"],
    "dpo_oob_tool": ["撤销刚才的操作", "把上一步撤回", "恢复刚删的东西"],
}


def build(n, seed=0):
    rng = random.Random(seed)
    # 按配比分配条数
    plan = []
    for fn, ratio in MATRIX:
        plan += [fn] * round(n * ratio)
    # 补齐/截断到 n
    while len(plan) < n:
        plan.append(MATRIX[0][0])
    plan = plan[:n]
    rng.shuffle(plan)

    out = []
    for i, fn in enumerate(plan):
        scene = scene_db.new_scene(seed * 100000 + 50000 + i)
        rec = fn(scene, scene.rng)
        # query 近义替换(prompt 最后一条 user),并与评测集去重
        # id_hallucination 的 query 与 rejected 的 unique_id 强绑定,跳过近义替换以保持自洽
        key = fn.__name__
        if key == "dpo_id_hallucination":
            variants = []
        else:
            variants = [v for v in PARAPHRASE.get(key, []) if _norm(v.replace("{n}", "")) not in EVAL_Q]
        if variants:
            v = rng.choice(variants)
            if "{n}" in v:
                v = v.format(n=rng.choice([2001, 2002, 2010, 2021]))
            if _norm(v) not in EVAL_Q:
                for m in rec["prompt"]:
                    if m["role"] == "user":
                        m["content"] = v
                        rec["meta"]["query"] = v
        rec["id"] = "dpo_{:05d}".format(i)
        # 训推一致(服务器大显存):每对样本携带全 22 完整工具(含 description),
        # 与推理端 eval_hf/runner 的 fc_eval/tools_ugc.json 同源同格式。
        rec["tools"] = toolset.FULL_TOOLS
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "dpo.jsonl"))
    a = ap.parse_args()
    rows = build(a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote {} DPO pairs -> {}".format(len(rows), a.out))


if __name__ == "__main__":
    main()
