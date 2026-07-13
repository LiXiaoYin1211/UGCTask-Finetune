# 训练数据统计报告（validate.py 自动生成）

## SFT（1000 条）— 校验通过

| 项 | 值 |
|----|----|
| 总数 | 1000 |
| 与评测集重复 | 0 |
| 校验错误 | 0 |
| 类别配比 | {'A': 854, 'B': 114, 'generic': 32} |
| A 内难度 | {'complex': 594, 'simple': 260} |
| B 子类(该问:不该问) | {'should_not_ask': 56, 'should_ask': 58} |

### 工具覆盖（升序）

| 工具 | 出现轨迹数 |
|------|-----------|
| ugc_analyze_scene_capture | 14 |
| ugc_get_indoor_info | 14 |
| ugc_get_house_info | 15 |
| ugc_query_paint_brushes | 15 |
| unreal_info | 23 |
| task | 42 |
| load_skill | 51 |
| ask_user | 58 |
| ugc_move_entity | 62 |
| ugc_paint_texture_surface | 63 |
| ugc_paint_color_surface | 64 |
| list_skills | 65 |
| ugc_get_scene_summary | 80 |
| ugc_delete_entity | 83 |
| ugc_query_paint_surfaces | 127 |
| ugc_place_on_surface | 135 |
| ugc_get_entity_detail | 135 |
| ugc_query_entities_by_area | 139 |
| ugc_create_entity | 160 |
| ugc_get_player_context | 201 |
| ugc_get_item_catalog | 267 |
| ugc_query_entities_by_type | 324 |

## DPO（800 对）— 校验通过

| 失败类型 | 对数 | 占比 |
|----------|------|------|
| should_ask | 176 | 22% |
| over_ask | 104 | 13% |
| id_hallucination | 160 | 20% |
| format_required | 56 | 7% |
| place_vs_create | 144 | 18% |
| skip_dependency | 120 | 15% |
| oob_tool | 40 | 5% |
| 与评测集重复 | 0 | - |
| 校验错误 | 0 | - |
