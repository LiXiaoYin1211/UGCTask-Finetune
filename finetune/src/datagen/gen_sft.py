"""gen_sft.py — 按配比生成 SFT 正例,输出 data/sft.jsonl。

配比(规格):A 65% / B 25% / 通用 10%;A 内 简单35/多步50/复杂15;B 内 该问:不该问=1:1。
每条轨迹由 prototypes 原型实例化 + 新场景采样,query 经近义扩展保证多样性。
禁 think:不写 reasoning_content。与 fc_eval/dataset.jsonl 的去重交给 validate.py。

用法: python gen_sft.py --n 1000 [--seed 0]
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


def _safe(q):
    """与评测集去重:命中则返回 None。"""
    return None if _norm(q) in EVAL_Q else q

# A-简单:单步查询类 query 模板(query, tool, args_fn, result_fn)
# 严格单步:一句话 -> 单一工具 -> 文本收尾。绝不调多余工具(不查场景/不查上下文)。
SIMPLE_SPECS = [
    ("这屋里现在有多少东西", "ugc_get_scene_summary", lambda s: {}, lambda s: s.scene_summary()),
    ("看看整个房子的结构", "ugc_get_house_info", lambda s: {}, lambda s: {"status": "success", "house_count": 1, "houses": []}),
    ("我现在站在哪", "ugc_get_player_context", lambda s: {}, lambda s: s.player_context()),
    ("当前引擎版本和项目名", "unreal_info", lambda s: {}, lambda s: {"engine_version": "5.3", "project_name": "UGCDemo", "is_editor": False}),
    ("列一下能用的 skill", "list_skills", lambda s: {}, lambda s: {"skills": [], "count": 0}),
    ("有哪些贴图笔刷", "ugc_query_paint_brushes", lambda s: {}, lambda s: s.paint_brushes()),
    ("哪些地板是室内的", "ugc_get_indoor_info", lambda s: {}, lambda s: {"status": "success", "indoor_floor_count": 1, "indoor_floors": []}),
    ("看看眼前这个场景什么风格", "ugc_analyze_scene_capture", lambda s: {"prompt": "这个场景什么装修风格"}, lambda s: {"status": "captured", "message": "Captured scene image."}),
    ("给我看下场景摘要", "ugc_get_scene_summary", lambda s: {}, lambda s: s.scene_summary()),
    ("获取一下玩家上下文", "ugc_get_player_context", lambda s: {}, lambda s: s.player_context()),
]
SIMPLE_PARAPHRASE = {
    "这屋里现在有多少东西": ["屋里现在有多少件东西啊", "统计下场景里有多少物件", "这房间总共多少东西"],
    "看看整个房子的结构": ["这房子都由什么组成", "看下整栋建筑结构", "房子的组成是什么"],
    "我现在站在哪": ["我现在在什么位置", "我站哪儿呢", "报下我的坐标和朝向"],
    "当前引擎版本和项目名": ["现在是什么引擎版本", "项目叫什么名字", "查下平台和项目信息"],
    "列一下能用的 skill": ["有哪些技能可用", "看下 skill 列表", "现在能用什么 skill"],
    "有哪些贴图笔刷": ["有什么纹理笔刷能用", "查下可用的贴图刷子", "列下贴图笔刷"],
    "哪些地板是室内的": ["哪些是室内地板", "看下室内外地板分布", "室内地板有哪些"],
    "看看眼前这个场景什么风格": ["眼前这块什么装修风格", "看下当前画面的风格", "帮我分析下这个场景风格"],
    "给我看下场景摘要": ["查看场景摘要", "输出当前场景摘要", "场景摘要给我看看"],
    "获取一下玩家上下文": ["取一下玩家上下文", "查下玩家上下文信息", "player context 给我"],
}

# B-该问:模糊请求 -> ask_user(措辞与评测集刻意区分,避免泄漏)
SHOULD_ASK_SPECS = [
    ("帮我装饰得漂亮些", "你想美化哪个区域？", ["客厅", "卧室", "整个家"]),
    ("整点有意思的布置", "你想要什么类型的内容呢？", ["摆点装饰", "搭个建筑", "其他"]),
    ("照着上回那样来一份", "你指的是上次的哪一项操作或物品？", ["最近放置的", "某个具体物品"]),
    ("给我换种装修感觉", "你想换成什么风格，针对哪个区域？", ["客厅", "整个家"]),
    ("帮我收拾布置一番", "你想布置哪个房间、什么用途？", ["客厅", "餐厅", "卧室"]),
    ("营造点氛围出来", "你希望在哪个区域营造什么氛围？", ["温馨客厅", "浪漫卧室"]),
    ("把它调整调整", "你想调整哪个物体、怎么调整？", ["位置", "大小", "颜色"]),
]
# B-不该问:可用工具自查(用 should_not 原型,措辞区分评测集)
SHOULD_NOT_QUERIES = [
    "我旁边都有哪些椅子", "我周边有什么家具", "我跟前摆着什么东西", "我身边有几把椅子",
    "看看我边上的家具",
]

GENERIC_CHAT = [
    ("你能帮我做什么", "我可以帮你在场景里检索、生成、编辑和染色物品。直接用口语告诉我你想做什么就行。"),
    ("谢谢你", "不客气，有需要随时叫我。"),
    ("你好", "你好，想对你的家园做点什么改造吗？"),
    ("这个怎么用啊", "你用自然语言下指令即可，比如“在桌上放个花瓶”或“把沙发挪到墙边”。"),
]


def gen_simple_query(scene, rng):
    """简单-单步查询类(8 个只读工具:场景摘要/玩家上下文/房屋/笔刷等)。"""
    spec = rng.choice(SIMPLE_SPECS)
    q, tool, args_fn, res_fn = spec
    pool = [x for x in ([q] + SIMPLE_PARAPHRASE.get(q, [])) if _safe(x)]
    q = rng.choice(pool) if pool else SIMPLE_PARAPHRASE.get(q, [q])[0]
    return P.proto_simple_query(scene, rng, q, tool, args_fn(scene), res_fn(scene))


def gen_simple(scene, rng):
    """简单类(单工具即返回 / 单步追问 / 闲聊)——占全库 35%。
    构成:单步查询 + 给定id单步create + 单步查目录 + 单步ask_user(该问) + 闲聊。"""
    r = rng.random()
    if r < 0.42:
        return gen_simple_query(scene, rng)                 # 单步只读查询
    elif r < 0.60:
        return P.proto_create_given_id(scene, rng)          # 给id+坐标 -> 单步 create
    elif r < 0.74:
        return P.proto_catalog_query(scene, rng)            # 纯查目录 -> 单步 catalog
    elif r < 0.90:
        return gen_should_ask(scene, rng)                   # 信息不足 -> 单步 ask_user
    else:
        return gen_generic(scene, rng)                      # 闲聊(0 工具)


def gen_task_complex(scene, rng):
    """复杂:先查场景摘要 -> 再 task 委派子助手(2 工具,多轮)。"""
    query = "帮我把整个客厅重新规划一下布局"
    a0, c0 = P.asst_call("ugc_get_scene_summary", {})
    a1, c1 = P.asst_call("task", {"description": "规划客厅布局",
                                  "prompt": "盘点客厅现有家具并给出重新布局方案，输出文字建议。",
                                  "subagent_type": "general"})
    msgs = [P.msg_system(), P.msg_user(query),
            a0, P.msg_tool(c0, scene.scene_summary()),
            a1, P.msg_tool(c1, {"summary": "建议：沙发靠北墙，茶几居中，书架沿西墙。"}),
            P.asst_text("我先盘点了客厅现有物件，子助手给出了布局建议（见上），需要我按此执行吗？")]
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "complex",
                                       "tools_used": ["ugc_get_scene_summary", "task"], "query": query}}


def gen_complex(scene, rng):
    """复杂类(多工具依赖链)——占全库 65%。
    覆盖全部写/查工具链;标杆:找桌子->查承载面->找花瓶ID->放置。"""
    protos = (
        [P.proto_place_chain] * 6 + [P.proto_filter_delete] * 4 +
        [P.proto_create_at_coord] * 3 + [P.proto_create_near_player] * 2 +
        [P.proto_paint_surface_chain] * 3 + [P.proto_paint_color_chain] * 3 +
        [P.proto_move_entity] * 3 + [P.proto_load_skill] * 2 +
        [gen_should_not_complex] * 3 + [gen_task_complex] * 2
    )
    return rng.choice(protos)(scene, rng)


def gen_should_not_complex(scene, rng):
    """复杂-不该问:'附近的椅子'用 player_context + query_by_area 自查(2 工具,防过度追问)。"""
    t = P.proto_ask_user_should_not(scene, rng)
    pool = [x for x in SHOULD_NOT_QUERIES if _safe(x)]
    if pool:
        t["messages"][1]["content"] = rng.choice(pool)
        t["meta"]["query"] = t["messages"][1]["content"]
    t["meta"]["difficulty"] = "complex"
    return t


def gen_should_ask(scene, rng):
    """单步追问:信息严重不足 -> ask_user(该问)。归入简单类。"""
    pool = [s for s in SHOULD_ASK_SPECS if _safe(s[0])]
    q, question, opts = rng.choice(pool)
    t = P.proto_ask_user_should(scene, rng, q, question, opts)
    t["meta"]["difficulty"] = "simple"
    return t


def gen_generic(scene, rng):
    """闲聊(0 工具)。归入简单类。"""
    q, a = rng.choice(GENERIC_CHAT)
    return {"messages": [P.msg_system(), P.msg_user(q), P.asst_text(a)],
            "meta": {"category": "generic", "difficulty": "simple", "tools_used": [], "query": q}}


def build(n, seed=0):
    """新分布:简单 35% / 复杂 65%。
    简单=单工具即返回 / 单步 ask_user / 闲聊;复杂=多工具依赖链。"""
    rng = random.Random(seed)
    n_simple = round(n * 0.35)
    n_complex = n - n_simple

    jobs = [gen_simple] * n_simple + [gen_complex] * n_complex
    rng.shuffle(jobs)

    out = []
    for i, fn in enumerate(jobs):
        scene = scene_db.new_scene(seed * 100000 + i)
        rec = fn(scene, scene.rng)
        rec["id"] = "sft_{:05d}".format(i)
        # 训推一致(服务器大显存):每条样本携带全 22 完整工具(含 description),
        # 与推理端 eval_hf/runner 用的 fc_eval/tools_ugc.json 完全同源同格式。
        rec["tools"] = toolset.FULL_TOOLS
        # 统一 difficulty 标签(原型内部值不一,按来源强制覆盖为 simple/complex 两类)
        rec["meta"]["difficulty"] = "simple" if fn is gen_simple else "complex"
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sft.jsonl"))
    a = ap.parse_args()
    rows = build(a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote {} SFT rows -> {}".format(len(rows), a.out))


if __name__ == "__main__":
    main()
