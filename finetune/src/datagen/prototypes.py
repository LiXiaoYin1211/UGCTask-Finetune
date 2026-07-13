"""prototypes.py — 教师模型(Claude-Opus-4.8)手写的标杆轨迹模板。

每个原型函数接收一个 Scene + rng,返回一条完整的 OpenAI messages 轨迹(SFT)
或一个分叉决策点(供 DPO 构造)。所有轨迹严格遵守 MyDataFiles/04:
- assistant 发起调用时 content=null, tool_calls 为数组, function.arguments 为字符串化 JSON;
- 每个 tool_call 有唯一 id, 对应 role:tool 消息 tool_call_id 一致;
- role 严格交替 assistant(tool_calls) -> tool(result) -> assistant ...;
- 禁 think: 不写 reasoning_content。
"""
import json
import scene_db

# 与 fc_eval/runner.py base SYSTEM_PROMPT 完全一致(训推一致)
SYSTEM_PROMPT = (
    "You are UnrealAIEngine, an in-game UGC building assistant running in the GameRuntime scope.\n"
    "You help the player build and edit their homestead by invoking tools in response to spoken instructions.\n\n"
    "# Core rules\n"
    "- Only call tools that have been advertised in this conversation. Never invent tool names.\n"
    "- NEVER fabricate or guess unique_id, item_table_id, surface names, or coordinates. Only use values returned by tools or given by the user.\n"
    "- item_table_id (which KIND of item, from the catalog) is NOT the same as unique_id (which specific PLACED instance). Do not mix them.\n"
    "- Before editing/deleting/moving a placed entity you MUST first query it (ugc_query_entities_by_type / _by_area / ugc_get_scene_summary) to obtain its unique_id.\n"
    "- To place an item onto an existing surface (floor/table/shelf/wall) you MUST use ugc_place_on_surface (query the target first). Do NOT use ugc_create_entity with a guessed coordinate for surface placement.\n"
    "- For vague spatial references (\"nearby\", \"in front of me\", \"by the lake\") first call ugc_get_player_context.\n"
    "- If a required decision, target, or value is missing and CANNOT be resolved with tools, you MUST call the ask_user tool. DO NOT ask clarifying questions in plain assistant text.\n"
    "- Be concise.\n\n"
    "# Tone\n"
    "Answer concisely. Take the correct first action toward the user's goal."
)

_CALL_SEQ = [0]


def _cid():
    _CALL_SEQ[0] += 1
    return "call_{:06d}".format(_CALL_SEQ[0])


# ---- 消息构造原语(强制格式正确) ----

def msg_system():
    return {"role": "system", "content": SYSTEM_PROMPT}


def msg_user(text):
    return {"role": "user", "content": text}


def asst_call(name, args):
    """assistant 发起一个工具调用; arguments 字符串化。返回 (message, call_id)。"""
    cid = _cid()
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args, ensure_ascii=False)}}]}, cid


def msg_tool(call_id, result_obj):
    return {"role": "tool", "tool_call_id": call_id,
            "content": json.dumps(result_obj, ensure_ascii=False)}


def asst_text(text):
    return {"role": "assistant", "content": text}


def asst_ask_user(question, options=None):
    cid = _cid()
    args = {"question": question}
    if options:
        args["options"] = options
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": "ask_user",
                                         "arguments": json.dumps(args, ensure_ascii=False)}}]}, cid


# ============ SFT 正例原型 ============
# 每个函数返回 {"messages":[...], "meta":{...}}; meta 记录 category/difficulty/tools_used/query

def proto_simple_query(scene, rng, query, tool_name, args, result):
    """A-简单:一句话 -> 单一查询工具 -> 文本收尾。"""
    a1, c1 = asst_call(tool_name, args)
    msgs = [msg_system(), msg_user(query), a1, msg_tool(c1, result),
            asst_text("（已为你查询，结果如上。）")]
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "simple",
                                       "tools_used": [tool_name], "query": query}}


def proto_place_chain(scene, rng):
    """A1 标杆:query_by_type -> get_entity_detail -> get_item_catalog -> place_on_surface 完整依赖链。"""
    vase_id = 2022
    msgs = [msg_system()]
    # 1 查桌子 —— 用查询返回的第一个实体作为后续目标,保证 ID 自洽
    qres = scene.query_by_type(2002)
    if not qres["entities"]:
        # 场景无桌子则退化为 filter_delete
        return proto_filter_delete(scene, rng)
    target = qres["entities"][0]
    region = next((e["region"] for e in scene.entities if e["unique_id"] == target["unique_id"]), "living_room")
    region_zh = scene_db.REGIONS[region]["zh"]
    query = "在{}那张桌子上摆个花瓶".format(region_zh)
    msgs.append(msg_user(query))
    a1, c1 = asst_call("ugc_query_entities_by_type", {"item_table_id": 2002})
    msgs += [a1, msg_tool(c1, qres)]
    # 2 拿 surface(用第一个实体的 unique_id)
    a2, c2 = asst_call("ugc_get_entity_detail", {"unique_id": target["unique_id"]})
    msgs += [a2, msg_tool(c2, scene.entity_detail(target["unique_id"]))]
    # 3 检索花瓶 item_table_id
    a3, c3 = asst_call("ugc_get_item_catalog", {"name": "vase", "limit": 5})
    msgs += [a3, msg_tool(c3, scene.catalog_search("vase", 5))]
    # 4 放置(target_entity_id 用 unique_id, items 用 item_table_id —— 二元论正确)
    a4, c4 = asst_call("ugc_place_on_surface",
                       {"target_entity_id": target["unique_id"], "surface_index": 0,
                        "items": [{"item_table_id": vase_id}]})
    msgs += [a4, msg_tool(c4, scene.place_result(target["unique_id"], [{"item_table_id": vase_id}]))]
    msgs.append(asst_text("已经把花瓶摆到{}的桌面上了。".format(region_zh)))
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "multi",
            "tools_used": ["ugc_query_entities_by_type", "ugc_get_entity_detail",
                           "ugc_get_item_catalog", "ugc_place_on_surface"], "query": query}}


def proto_filter_delete(scene, rng):
    """A2 标杆:player_context -> query_by_area -> 语义过滤"小的" -> 逐个 delete。"""
    query = "把湖边那些小盆栽删掉"
    center = scene_db.REGIONS["lakeside"]["center"]
    msgs = [msg_system(), msg_user(query)]
    a1, c1 = asst_call("ugc_get_player_context", {})
    msgs += [a1, msg_tool(c1, scene.player_context())]
    a2, c2 = asst_call("ugc_query_entities_by_area", {"center": center, "radius": 600, "item_table_id": 2018})
    area = scene.query_by_area(center, 600, 2018)
    msgs += [a2, msg_tool(c2, area)]
    targets = [e["unique_id"] for e in area["entities"][:2]]
    for uid in targets:
        a, c = asst_call("ugc_delete_entity", {"unique_id": uid})
        msgs += [a, msg_tool(c, scene.delete_result(uid))]
    msgs.append(asst_text("已删除湖边 {} 个小盆栽。".format(len(targets))))
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "multi",
            "tools_used": ["ugc_get_player_context", "ugc_query_entities_by_area", "ugc_delete_entity"],
            "query": query}}


def proto_ask_user_should(scene, rng, query, question, options):
    """B1 标杆:信息严重不足 -> 第一步就 ask_user。"""
    a1, c1 = asst_ask_user(question, options)
    msgs = [msg_system(), msg_user(query), a1,
            msg_tool(c1, {"answer": options[0] if options else "（用户给出澄清）"}),
            asst_text("明白了，这就为你处理。")]
    return {"messages": msgs, "meta": {"category": "B", "subtype": "should_ask",
                                       "tools_used": ["ask_user"], "query": query}}


def proto_ask_user_should_not(scene, rng):
    """B2 标杆:"附近的椅子" -> 用 player_context 自查、不问(防过度触发)。"""
    query = "我附近有哪些椅子"
    msgs = [msg_system(), msg_user(query)]
    a1, c1 = asst_call("ugc_get_player_context", {})
    pc = scene.player_context()
    msgs += [a1, msg_tool(c1, pc)]
    a2, c2 = asst_call("ugc_query_entities_by_area",
                       {"center": pc["nearby_center"], "radius": 400, "item_table_id": 2001})
    msgs += [a2, msg_tool(c2, scene.query_by_area(pc["nearby_center"], 400, 2001))]
    msgs.append(asst_text("你附近的椅子已经列出。"))
    return {"messages": msgs, "meta": {"category": "B", "subtype": "should_not_ask",
            "tools_used": ["ugc_get_player_context", "ugc_query_entities_by_area"], "query": query}}


def proto_paint_surface_chain(scene, rng):
    """长尾工具覆盖:query_by_type -> query_paint_surfaces(拿surface_name) -> paint_texture/color。
    治 paint_surfaces 训练 0 覆盖 + paint 类工具依赖前置查询。"""
    # 找一个可染色实体(墙/家具)
    qres = scene.query_by_type(1002)  # HouseWall
    if not qres["entities"]:
        qres = scene.query_by_type(2004)  # Table 兜底
    if not qres["entities"]:
        return proto_simple_query(scene, rng, "看看有哪些贴图笔刷",
                                  "ugc_query_paint_brushes", {}, scene.paint_brushes())
    target = qres["entities"][0]
    uid = target["unique_id"]
    itid = qres["entities"][0].get("item_table_id", 1002)
    query = rng.choice(["给那面墙刷个木纹", "把这面墙的贴图换一下", "给墙面上个纹理"])
    msgs = [msg_system(), msg_user(query)]
    a1, c1 = asst_call("ugc_query_entities_by_type", {"item_table_id": itid})
    msgs += [a1, msg_tool(c1, qres)]
    # 查该实体的可染色面(paint_surfaces 只需 unique_id —— 正确教法)
    a2, c2 = asst_call("ugc_query_paint_surfaces", {"unique_id": uid})
    ps = scene.paint_surfaces(uid)
    msgs += [a2, msg_tool(c2, ps)]
    sname = ps["surfaces"][0]["surface_name"]
    brush = "brush_wood_01"
    a3, c3 = asst_call("ugc_paint_texture_surface",
                       {"unique_id": uid, "surface_name": sname, "texture_brush_id": brush})
    msgs += [a3, msg_tool(c3, {"status": "success", "unique_id": uid, "surface_name": sname})]
    msgs.append(asst_text("已给这面墙刷上木纹贴图。"))
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "multi",
            "tools_used": ["ugc_query_entities_by_type", "ugc_query_paint_surfaces",
                           "ugc_paint_texture_surface"], "query": query}}


def proto_paint_color_chain(scene, rng):
    """长尾:query_by_type -> query_paint_surfaces -> paint_color_surface。"""
    qres = scene.query_by_type(1002)
    if not qres["entities"]:
        qres = scene.query_by_type(2015)  # Sofa 兜底
    if not qres["entities"]:
        return proto_paint_surface_chain(scene, rng)
    target = qres["entities"][0]
    uid = target["unique_id"]
    itid = qres["entities"][0].get("item_table_id", 1002)
    query = rng.choice(["把那面墙刷成蓝色", "给墙面换个颜色", "这面墙涂成米白"])
    msgs = [msg_system(), msg_user(query)]
    a1, c1 = asst_call("ugc_query_entities_by_type", {"item_table_id": itid})
    msgs += [a1, msg_tool(c1, qres)]
    a2, c2 = asst_call("ugc_query_paint_surfaces", {"unique_id": uid})
    ps = scene.paint_surfaces(uid)
    msgs += [a2, msg_tool(c2, ps)]
    sname = ps["surfaces"][0]["surface_name"]
    a3, c3 = asst_call("ugc_paint_color_surface",
                       {"unique_id": uid, "surface_name": sname,
                        "color_params": {"custom_color1": "#3A6EA5"}})
    msgs += [a3, msg_tool(c3, {"status": "success", "unique_id": uid, "surface_name": sname})]
    msgs.append(asst_text("已把墙面刷成指定颜色。"))
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "multi",
            "tools_used": ["ugc_query_entities_by_type", "ugc_query_paint_surfaces",
                           "ugc_paint_color_surface"], "query": query}}


def proto_create_given_id(scene, rng):
    """简单(单步):用户直接给 item_table_id + 坐标 -> 单步 ugc_create_entity。
    id 已知,无需查目录 -> 只调一个工具。对齐用户示例2。"""
    itid = rng.choice([2001, 2002, 2015, 2018, 2021, 2022])
    x = rng.choice([100, 200, -150, 300, -80, 250, 0])
    y = rng.choice([100, -120, 50, 200, -60, 180, 0])
    z = rng.choice([0, -50, 75, 0])
    tmpl = rng.choice([
        "在（{x}，{y}，{z}）位置放置 id={i} 的实体",
        "在坐标 ({x}, {y}, {z}) 创建 item_table_id={i} 的物品",
        "把 id 为 {i} 的实体放到 {x},{y},{z}",
    ])
    query = tmpl.format(x=x, y=y, z=z, i=itid)
    loc = {"x": x, "y": y, "z": z}
    a1, c1 = asst_call("ugc_create_entity", {"item_table_id": itid, "location": loc})
    msgs = [msg_system(), msg_user(query), a1, msg_tool(c1, scene.create_result(itid, loc)),
            asst_text("已在 ({x}, {y}, {z}) 创建实体（item_table_id: {i}）。".format(x=x, y=y, z=z, i=itid))]
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "simple",
            "tools_used": ["ugc_create_entity"], "query": query}}


def proto_catalog_query(scene, rng):
    """简单(单步):纯目录检索 -> 单步 ugc_get_item_catalog。对齐用户示例3。"""
    kind, en = rng.choice([("沙发", "sofa"), ("花瓶", "vase"), ("台灯", "lamp"),
                           ("椅子", "chair"), ("桌子", "table"), ("盆栽", "plant"), ("书架", "shelves")])
    tmpl = rng.choice([
        "查找 UGC 实体：{k}",
        "目录里有没有{k}",
        "搜一下{k}的 item_table_id",
        "帮我在物品目录查{k}",
    ])
    query = tmpl.format(k=kind)
    a1, c1 = asst_call("ugc_get_item_catalog", {"name": en, "limit": 10})
    msgs = [msg_system(), msg_user(query), a1, msg_tool(c1, scene.catalog_search(en, 10)),
            asst_text("已为你在目录中检索{k}，结果如上。".format(k=kind))]
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "simple",
            "tools_used": ["ugc_get_item_catalog"], "query": query}}


def proto_create_at_coord(scene, rng):
    """精确坐标放置(用户已给坐标):get_item_catalog(拿 item_table_id) -> create_entity。
    关键:用户给了明确坐标就无需 player_context;但 item_table_id 必须查目录得到,不得编造。
    不多(不调 player_context/query)、不漏(必须先 catalog 再 create)。"""
    kind, en = rng.choice([("沙发", "sofa"), ("桌子", "table"), ("台灯", "lamp"),
                           ("花瓶", "vase"), ("椅子", "chair"), ("盆栽", "plant")])
    x = rng.choice([100, 200, -150, 300, -80, 250])
    y = rng.choice([100, -120, 50, 200, -60, 180])
    z = rng.choice([0, -50, 75, 0, 0])
    tmpl = rng.choice([
        "在（{x},{y},{z}）位置放置一个{k}",
        "在坐标 ({x}, {y}, {z}) 放个{k}",
        "帮我在 {x},{y},{z} 这个位置创建一个{k}",
    ])
    query = tmpl.format(x=x, y=y, z=z, k=kind)
    msgs = [msg_system(), msg_user(query)]
    a1, c1 = asst_call("ugc_get_item_catalog", {"name": en, "limit": 10})
    cat = scene.catalog_search(en, 10)
    msgs += [a1, msg_tool(c1, cat)]
    itid = cat["items"][0]["item_table_id"] if cat["items"] else 2
    loc = {"x": x, "y": y, "z": z}
    a2, c2 = asst_call("ugc_create_entity", {"item_table_id": itid, "location": loc})
    msgs += [a2, msg_tool(c2, scene.create_result(itid, loc))]
    msgs.append(asst_text("已在 ({x}, {y}, {z}) 放置{k}（item_table_id: {i}）。".format(
        x=x, y=y, z=z, k=kind, i=itid)))
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "multi",
            "tools_used": ["ugc_get_item_catalog", "ugc_create_entity"], "query": query}}


def proto_create_near_player(scene, rng):
    """模糊位置放置(在我面前/这儿):player_context(拿坐标) -> get_item_catalog -> create_entity。
    三步都必要:模糊空间需 player_context;item_table_id 需 catalog;最后 create。不多不漏。"""
    kind, en = rng.choice([("桌子", "table"), ("沙发", "sofa"), ("盆栽", "plant"), ("台灯", "lamp")])
    query = rng.choice(["在我面前放张{k}", "这儿摆个{k}", "在我旁边放个{k}"]).format(k=kind)
    pc = scene.player_context()
    msgs = [msg_system(), msg_user(query)]
    a1, c1 = asst_call("ugc_get_player_context", {})
    msgs += [a1, msg_tool(c1, pc)]
    a2, c2 = asst_call("ugc_get_item_catalog", {"name": en, "limit": 10})
    cat = scene.catalog_search(en, 10)
    msgs += [a2, msg_tool(c2, cat)]
    itid = cat["items"][0]["item_table_id"] if cat["items"] else 2
    loc = dict(pc.get("nearby_center") or {"x": 0, "y": 0, "z": 0})
    a3, c3 = asst_call("ugc_create_entity", {"item_table_id": itid, "location": loc})
    msgs += [a3, msg_tool(c3, scene.create_result(itid, loc))]
    msgs.append(asst_text("已在你附近放置{k}。".format(k=kind)))
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "multi",
            "tools_used": ["ugc_get_player_context", "ugc_get_item_catalog", "ugc_create_entity"],
            "query": query}}


def proto_create_entity(scene, rng):
    """兼容旧名:随机走坐标放置或模糊放置之一。"""
    return rng.choice([proto_create_at_coord, proto_create_near_player])(scene, rng)


def proto_move_entity(scene, rng):
    """长尾:query_by_type -> get_entity_detail -> move_entity(先查再移)。"""
    qres = scene.query_by_type(2015)  # Sofa
    if not qres["entities"]:
        qres = scene.query_by_type(2004)
    if not qres["entities"]:
        return proto_create_entity(scene, rng)
    uid = qres["entities"][0]["unique_id"]
    itid = qres["entities"][0].get("item_table_id", 2015)
    new_loc = {"x": 300, "y": -120, "z": 0}
    query = rng.choice(["把沙发挪到墙边", "沙发往北移一点", "把那个沙发换个位置"])
    msgs = [msg_system(), msg_user(query)]
    a1, c1 = asst_call("ugc_query_entities_by_type", {"item_table_id": itid})
    msgs += [a1, msg_tool(c1, qres)]
    a2, c2 = asst_call("ugc_move_entity", {"unique_id": uid, "new_location": new_loc})
    msgs += [a2, msg_tool(c2, scene.move_result(uid, new_loc))]
    msgs.append(asst_text("已把沙发移到新位置。"))
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "multi",
            "tools_used": ["ugc_query_entities_by_type", "ugc_move_entity"], "query": query}}


def proto_load_skill(scene, rng):
    """长尾:list_skills -> load_skill(先列再加载)。"""
    query = rng.choice(["加载一下建房子的技能", "启用那个批量摆放的 skill", "把装修技能加载进来"])
    msgs = [msg_system(), msg_user(query)]
    a1, c1 = asst_call("list_skills", {})
    msgs += [a1, msg_tool(c1, {"skills": [{"name": "build_house"}, {"name": "batch_place"}], "count": 2})]
    a2, c2 = asst_call("load_skill", {"name": "build_house"})
    msgs += [a2, msg_tool(c2, {"status": "success", "loaded": "build_house"})]
    msgs.append(asst_text("技能已加载，可以用了。"))
    return {"messages": msgs, "meta": {"category": "A", "difficulty": "multi",
            "tools_used": ["list_skills", "load_skill"], "query": query}}


# ============ DPO 分叉原型 ============
# 返回 {"prompt":[...], "chosen":[...], "rejected":[...], "meta":{...}}
# prompt 为共享前缀(到分叉前); chosen/rejected 从分叉点开始。

def dpo_should_ask(scene, rng):
    """该问不问:模糊请求 -> chosen=ask_user, rejected=瞎猜执行。"""
    query = "把那个弄好看点"
    prompt = [msg_system(), msg_user(query)]
    chosen, _ = asst_ask_user("你想美化哪个区域或物体？", ["客厅", "整个家", "某件家具"])
    bad, bc = asst_call("ugc_paint_color_surface",
                        {"unique_id": 9999, "surface_name": "default",
                         "color_params": {"custom_color1": "#FF8800"}})
    rejected = [bad, msg_tool(bc, {"error": "Entity not found: unique_id=9999"})]
    return {"prompt": prompt, "chosen": [chosen], "rejected": rejected,
            "meta": {"type": "should_ask", "query": query, "chosen_tools": ["ask_user"]}}


def dpo_id_hallucination(scene, rng):
    """ID 幻觉:用户给 item_table_id, chosen=先查拿 unique_id, rejected=把 item_table_id 当 unique_id 直接删。"""
    itid = rng.choice([2001, 2002, 2010, 2021])
    # 措辞与评测集(如"把 id 2002 的那张桌子删掉")刻意区分,避免泄漏
    tmpl = rng.choice([
        "编号 {n} 的家具帮我移除掉",
        "把物品 {n} 这一类的都清掉",
        "删一下配置 id 为 {n} 的那个家具",
        "id={n} 的家具不想要了，去掉",
    ])
    query = tmpl.format(n=itid)
    prompt = [msg_system(), msg_user(query)]
    good, gc = asst_call("ugc_query_entities_by_type", {"item_table_id": itid})
    chosen = [good, msg_tool(gc, scene.query_by_type(itid))]
    bad, bc = asst_call("ugc_delete_entity", {"unique_id": itid})  # 把 item_table_id 当 unique_id
    rejected = [bad, msg_tool(bc, {"error": "Entity not found: unique_id={}".format(itid)})]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "meta": {"type": "id_hallucination", "query": query, "item_table_id": itid, "chosen_tools": ["ugc_query_entities_by_type"]}}


def dpo_place_vs_create(scene, rng):
    """错选工具:在桌上摆东西, chosen=先查再 place, rejected=直接 create 猜坐标。"""
    query = "在桌子上放个台灯"
    prompt = [msg_system(), msg_user(query)]
    good, gc = asst_call("ugc_query_entities_by_type", {"item_table_id": 2002})
    chosen = [good, msg_tool(gc, scene.query_by_type(2002))]
    bad, bc = asst_call("ugc_create_entity",
                        {"item_table_id": 2015, "location": {"x": 100, "y": 50, "z": 75}})
    rejected = [bad, msg_tool(bc, scene.create_result(2015, {"x": 100, "y": 50, "z": 75}))]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "meta": {"type": "place_vs_create", "query": query, "chosen_tools": ["ugc_query_entities_by_type"]}}


def dpo_skip_dependency(scene, rng):
    """跳过前置依赖:染色, chosen=先 query_paint_surfaces, rejected=直接 paint 编 surface_name。"""
    ent = scene.pick_entity(item_table_id=1002) or scene.pick_entity()
    uid = ent["unique_id"]
    query = "给那面墙刷成蓝色"
    prompt = [msg_system(), msg_user(query)]
    good, gc = asst_call("ugc_query_entities_by_type", {"item_table_id": 1002})
    chosen = [good, msg_tool(gc, scene.query_by_type(1002))]
    bad, bc = asst_call("ugc_paint_color_surface",
                        {"unique_id": uid, "surface_name": "main",
                         "color_params": {"custom_color1": "#0000FF"}})
    rejected = [bad, msg_tool(bc, {"error": "surface 'main' not found; query paint surfaces first"})]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "meta": {"type": "skip_dependency", "query": query, "chosen_tools": ["ugc_query_entities_by_type"]}}


def dpo_over_ask(scene, rng):
    """过度追问(反向):"附近的椅子"本可自查, chosen=player_context自查, rejected=ask_user 过度追问。"""
    query = "我附近有什么椅子"
    prompt = [msg_system(), msg_user(query)]
    good, gc = asst_call("ugc_get_player_context", {})
    chosen = [good, msg_tool(gc, scene.player_context())]
    bad, bc = asst_ask_user("请问“附近”是指多大范围？", ["1米", "5米", "整个房间"])
    rejected = [bad, msg_tool(bc, {"answer": "5米"})]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "meta": {"type": "over_ask", "query": query, "chosen_tools": ["ugc_get_player_context"]}}


def dpo_format_required(scene, rng):
    """格式/必填:检索花瓶, chosen=name+limit, rejected=给name漏limit。"""
    query = "目录里有哪些花瓶"
    prompt = [msg_system(), msg_user(query)]
    good, gc = asst_call("ugc_get_item_catalog", {"name": "vase", "limit": 5})
    chosen = [good, msg_tool(gc, scene.catalog_search("vase", 5))]
    bad, bc = asst_call("ugc_get_item_catalog", {"name": "vase"})  # 漏 limit
    rejected = [bad, msg_tool(bc, {"error": "The 'limit' parameter is required when 'name' is provided."})]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "meta": {"type": "format_required", "query": query, "chosen_tools": ["ugc_get_item_catalog"]}}


def dpo_oob_tool(scene, rng):
    """越界/幻觉工具名:chosen=用已注册工具, rejected=调编造的工具名。"""
    query = "撤销刚才的操作"
    prompt = [msg_system(), msg_user(query)]
    chosen = [asst_text("运行时没有 AI 撤销工具；如需恢复，我可以用反向操作（例如重新创建被删对象）。请告诉我要恢复什么。")]
    bad, bc = asst_call("ugc_undo", {})  # 不存在的工具
    rejected = [bad, msg_tool(bc, {"error": "Tool 'ugc_undo' is not available."})]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "meta": {"type": "oob_tool", "query": query, "chosen_tools": []}}
