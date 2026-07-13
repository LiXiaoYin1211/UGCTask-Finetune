"""scene_db.py — 仿真场景库,为训练数据生成提供"ID 二元论一致"的采样 API。

核心保证(代码兜底,LLM 最易错的部分):
- item_table_id = 配置ID(哪一种),来自 ../fc_eval/item_table.json
- unique_id     = 实例ID(哪一个),场景中已放置实体的唯一标识
- 二者绝不混用;查询类工具返回 unique_id,写操作消费 unique_id。

提供:目录采样、按类型/区域查实体、surface、player_context、笔刷等,
均返回与 ../MyDataFiles/06 对齐的 JSON 结构,可直接作为 role:tool 的 content。
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # finetune/ 根
ITEM_TABLE_PATH = os.path.join(ROOT, "..", "fc_eval", "item_table.json")


def _load_catalog():
    data = json.load(open(ITEM_TABLE_PATH, encoding="utf-8"))
    items = []
    for group in ("primitives", "house", "furniture", "voxel_building"):
        for it in data.get(group, []):
            items.append({
                "item_table_id": it["item_table_id"],
                "display_name": it["name"],
                "category": "furniture",  # 运行时目录统一归 furniture/homestead,仿真用 furniture
                "keywords": it.get("keywords", []),
                "group": group,
            })
    return items


CATALOG = _load_catalog()
CATALOG_BY_ID = {c["item_table_id"]: c for c in CATALOG}

# 语义关键词 -> item_table_id 列表(供"找椅子/灯/花瓶"类检索)
KEYWORD_INDEX = {}
for c in CATALOG:
    for kw in c["keywords"]:
        KEYWORD_INDEX.setdefault(kw.lower(), []).append(c["item_table_id"])

# 典型 bounds(cm),用于"大/小"语义过滤与 place 的尺寸校验(仿真值)
BOUNDS = {
    2001: (45, 45, 95), 2002: (120, 80, 75), 2010: (90, 90, 45),
    2015: (20, 20, 45), 2018: (30, 30, 55), 2019: (100, 30, 180),
    2021: (200, 90, 85), 2022: (25, 25, 40), 2016: (60, 3, 40),
    1001: (400, 400, 20), 1002: (400, 20, 300),
}

# 区域定义(坐标聚类),支撑"客厅/餐厅/湖边/附近"
REGIONS = {
    "living_room": {"zh": "客厅", "center": {"x": 0, "y": 0, "z": 0}, "radius": 300, "indoor": True},
    "dining_room": {"zh": "餐厅", "center": {"x": 400, "y": 200, "z": 0}, "radius": 250, "indoor": True},
    "lakeside": {"zh": "湖边", "center": {"x": 2000, "y": 1500, "z": -50}, "radius": 600, "indoor": False},
}

# 染色笔刷(仿真)
TEXTURE_BRUSHES = [
    {"texture_brush_id": "WoodOak", "description": "橡木纹", "asset": "/Game/Brush/Brush_WoodOak"},
    {"texture_brush_id": "MarbleWhite", "description": "白大理石", "asset": "/Game/Brush/Brush_MarbleWhite"},
    {"texture_brush_id": "FabricLinen", "description": "亚麻布纹", "asset": "/Game/Brush/Brush_FabricLinen"},
]


class Scene:
    """一个可采样的仿真场景实例。每次生成一条轨迹时新建,保证 unique_id 自洽。"""

    def __init__(self, rng):
        self.rng = rng
        self._uid = rng.randint(5000, 8000)
        self.entities = []        # 已放置实体列表(每个含 unique_id/item_table_id/location/region)
        self.player = None
        self._populate()

    def _next_uid(self):
        self._uid += 1
        return self._uid

    def _populate(self):
        """随机铺一个家:客厅/餐厅家具 + 湖边自然物。"""
        rng = self.rng
        plan = [
            ("living_room", 1001, 1), ("living_room", 1002, 2),
            ("living_room", 2002, 1), ("living_room", 2021, 1),
            ("living_room", 2019, 1), ("living_room", 2015, 1),
            ("dining_room", 2002, 1), ("dining_room", 2001, 4),
            ("lakeside", 2018, 3), ("lakeside", 2022, 0),
        ]
        for region, itid, n in plan:
            for _ in range(n):
                self._place(region, itid)
        # player 站在客厅附近
        self.player = {
            "location": {"x": 60, "y": 40, "z": 0},
            "rotation": {"pitch": 0, "yaw": 90, "roll": 0},
            "forward": {"x": 0, "y": 1, "z": 0},
            "nearby_center": {"x": 60, "y": 40, "z": 0},
            "facing_source": "camera_yaw",
        }

    def _place(self, region, item_table_id):
        rng = self.rng
        c = REGIONS[region]["center"]
        loc = {"x": c["x"] + rng.randint(-120, 120),
               "y": c["y"] + rng.randint(-120, 120),
               "z": 0}
        self._uid += 1
        ent = {
            "unique_id": self._uid,
            "item_table_id": item_table_id,
            "display_name": CATALOG_BY_ID[item_table_id]["display_name"],
            "location": loc,
            "rotation": {"pitch": 0, "yaw": rng.choice([0, 90, 180, 270]), "roll": 0},
            "region": region,
        }
        self.entities.append(ent)
        return ent

    # ---- 采样 / 查询 API(返回值对齐工具真实 JSON) ----

    def catalog_search(self, name, limit):
        """ugc_get_item_catalog(name, limit) 的仿真返回。"""
        kw = (name or "").lower()
        hits = []
        for itid in KEYWORD_INDEX.get(kw, []):
            c = CATALOG_BY_ID[itid]
            hits.append({"item_table_id": itid, "display_name": c["display_name"],
                         "category": "furniture", "similarity": round(self.rng.uniform(0.82, 0.96), 2)})
        if not hits:  # 模糊回退:display_name 子串匹配
            for c in CATALOG:
                if kw and kw in c["display_name"].lower():
                    hits.append({"item_table_id": c["item_table_id"], "display_name": c["display_name"],
                                 "category": "furniture", "similarity": round(self.rng.uniform(0.6, 0.8), 2)})
        hits = hits[:limit]
        mode = "semantic" if hits and hits[0]["similarity"] >= 0.82 else "fuzzy_fallback"
        return {"items": hits, "returned_count": len(hits), "matched_count": len(hits),
                "total_available_count": len(CATALOG), "search_mode": mode,
                "semantic_search_used": mode == "semantic"}

    def query_by_type(self, item_table_id, limit=50):
        ents = [self._entity_json(e) for e in self.entities if e["item_table_id"] == item_table_id]
        ents = ents[:limit]
        out = {"entities": ents, "total_count": len(ents), "is_truncated": False}
        if not ents:
            out["message"] = "No entities of item_table_id={} found.".format(item_table_id)
        return out

    def query_by_area(self, center, radius, item_table_id=None):
        res = []
        for e in self.entities:
            if item_table_id and e["item_table_id"] != item_table_id:
                continue
            d = ((e["location"]["x"] - center["x"]) ** 2 +
                 (e["location"]["y"] - center["y"]) ** 2 +
                 (e["location"]["z"] - center["z"]) ** 2) ** 0.5
            if d <= radius:
                j = self._entity_json(e)
                j["distance"] = round(d, 1)
                res.append(j)
        res.sort(key=lambda x: x["distance"])
        return {"level": "entity", "entities": res, "total_count": len(res), "is_truncated": False}

    def entity_detail(self, unique_id):
        e = self._find(unique_id)
        if not e:
            return {"error": "Entity not found: unique_id={}".format(unique_id)}
        j = self._entity_json(e)
        j["scale"] = {"x": 1, "y": 1, "z": 1}
        j["category"] = "furniture"
        j["surfaces"] = [{"index": 0, "is_above": True, "surface_type": 3,
                          "surface_type_description": "top", "size": {"x": 80, "y": 60}}]
        return j

    def paint_surfaces(self, unique_id):
        e = self._find(unique_id)
        if not e:
            return {"error": "Entity not found: unique_id={}".format(unique_id)}
        return {"status": "success", "unique_id": unique_id, "item_table_id": e["item_table_id"],
                "surfaces_count": 1,
                "surfaces": [{"surface_name": "default", "texture_brush_id": "",
                              "color_params": {"custom_color1": {"hex": "#FFFFFF"}}}]}

    def paint_brushes(self):
        return {"status": "success", "texture_brushes_count": len(TEXTURE_BRUSHES),
                "texture_brushes": TEXTURE_BRUSHES,
                "color_brush_schema": {"custom_color1": "#RRGGBB"}}

    def scene_summary(self):
        from collections import Counter
        cnt = Counter(e["item_table_id"] for e in self.entities)
        summary = [{"item_table_id": k, "display_name": CATALOG_BY_ID[k]["display_name"], "count": v}
                   for k, v in cnt.items()]
        return {"summary": summary, "type_count": len(cnt),
                "total_entity_count": len(self.entities), "player_gid": 1, "homeland_id": 1}

    def player_context(self):
        return dict(self.player, world="RuntimeWorld",
                    usage_hint="Use nearby_center for vague spatial needs.")

    def create_result(self, item_table_id, location):
        return {"status": "command_submitted", "item_table_id": item_table_id,
                "location": location, "message": "Entity creation submitted. Query later to get unique_id."}

    def place_result(self, target_entity_id, items):
        out = []
        for it in items:
            out.append({"status": "command_submitted", "item_table_id": it["item_table_id"],
                        "display_name": CATALOG_BY_ID.get(it["item_table_id"], {}).get("display_name", "?"),
                        "actual_location": {"x": 0, "y": 0, "z": 75}, "message": "Placed on surface."})
        return {"results": out}

    def delete_result(self, unique_id):
        e = self._find(unique_id)
        if not e:
            return {"error": "Entity not found: unique_id={}".format(unique_id)}
        return {"status": "success", "deleted_unique_id": unique_id,
                "item_table_id": e["item_table_id"], "last_location": e["location"]}

    def move_result(self, unique_id, new_location=None, new_rotation=None):
        e = self._find(unique_id)
        if not e:
            return {"error": "Entity not found: unique_id={}".format(unique_id)}
        return {"status": "success", "unique_id": unique_id,
                "location": new_location or e["location"],
                "rotation": new_rotation or e["rotation"], "scale": {"x": 1, "y": 1, "z": 1}}

    # ---- 内部工具 ----
    def _find(self, unique_id):
        for e in self.entities:
            if e["unique_id"] == unique_id:
                return e
        return None

    def _entity_json(self, e):
        return {"unique_id": e["unique_id"], "item_table_id": e["item_table_id"],
                "display_name": e["display_name"], "location": e["location"],
                "rotation": e["rotation"]}

    def pick_entity(self, region=None, item_table_id=None):
        pool = [e for e in self.entities
                if (region is None or e["region"] == region)
                and (item_table_id is None or e["item_table_id"] == item_table_id)]
        return self.rng.choice(pool) if pool else None

    def pick_catalog(self, group=None):
        pool = [c for c in CATALOG if group is None or c["group"] == group]
        return self.rng.choice(pool)


def new_scene(seed=None):
    return Scene(random.Random(seed))


if __name__ == "__main__":
    s = new_scene(42)
    print("entities:", len(s.entities))
    print("sample by type 2002:", json.dumps(s.query_by_type(2002), ensure_ascii=False)[:200])
    print("catalog vase:", json.dumps(s.catalog_search("vase", 5), ensure_ascii=False)[:200])
