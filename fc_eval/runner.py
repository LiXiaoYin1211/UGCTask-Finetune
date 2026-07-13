"""
runner.py — 调用本地 Ollama qwen3.5:2b,对每条评测用例在 think 开/关两种模式下各跑一次,
落原始响应到 results/raw_<mode>.jsonl。

用法:
  <venv_python> runner.py [--model qwen3.5:2b] [--limit N]
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

from common import results_dir

HERE = os.path.dirname(os.path.abspath(__file__))
API_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:2b"

# 系统提示:精简自 MyDataFiles/02_系统约定与能力边界.md 的 base prompt,
# 强调 GameRuntime scope、追问走 ask_user、不编造 ID、先查询后写操作。
SYSTEM_PROMPT = """You are UnrealAIEngine, an in-game UGC building assistant running in the GameRuntime scope.
You help the player build and edit their homestead by invoking tools in response to spoken instructions.

# Core rules
- Only call tools that have been advertised in this conversation. Never invent tool names.
- NEVER fabricate or guess unique_id, item_table_id, surface names, or coordinates. Only use values returned by tools or given by the user.
- item_table_id (which KIND of item, from the catalog) is NOT the same as unique_id (which specific PLACED instance). Do not mix them.
- Before editing/deleting/moving a placed entity you MUST first query it (ugc_query_entities_by_type / _by_area / ugc_get_scene_summary) to obtain its unique_id.
- To place an item onto an existing surface (floor/table/shelf/wall) you MUST use ugc_place_on_surface (query the target first). Do NOT use ugc_create_entity with a guessed coordinate for surface placement.
- For vague spatial references ("nearby", "in front of me", "by the lake") first call ugc_get_player_context.
- If a required decision, target, or value is missing and CANNOT be resolved with tools, you MUST call the ask_user tool. DO NOT ask clarifying questions in plain assistant text.
- Be concise.

# Tone
Answer concisely. Take the correct first action toward the user's goal."""

# 提示词消融变体:在 base 之上"轻度强化"ask_user 的优先级(只动追问相关的指引),
# 用于验证"C 类崩溃是否因 prompt 没把 ask_user 讲到位"。其余约束与 base 完全一致。
SYSTEM_PROMPT_ASKUSER = """You are UnrealAIEngine, an in-game UGC building assistant running in the GameRuntime scope.
You help the player build and edit their homestead by invoking tools in response to spoken instructions.

# Core rules
- Only call tools that have been advertised in this conversation. Never invent tool names.
- NEVER fabricate or guess unique_id, item_table_id, surface names, or coordinates. Only use values returned by tools or given by the user.
- item_table_id (which KIND of item, from the catalog) is NOT the same as unique_id (which specific PLACED instance). Do not mix them.
- Before editing/deleting/moving a placed entity you MUST first query it (ugc_query_entities_by_type / _by_area / ugc_get_scene_summary) to obtain its unique_id.
- To place an item onto an existing surface (floor/table/shelf/wall) you MUST use ugc_place_on_surface (query the target first). Do NOT use ugc_create_entity with a guessed coordinate for surface placement.
- For vague spatial references ("nearby", "in front of me", "by the lake") first call ugc_get_player_context.
- Be concise.

# Clarify-first rule (HIGHEST PRIORITY)
When the user's instruction is ambiguous about WHAT object to act on, or WHAT the desired outcome is, and the answer CANNOT be uniquely determined from the request itself, your FIRST action MUST be to call the `ask_user` tool to ask one clarifying question.
- Do NOT call a query tool (e.g. ugc_get_player_context / ugc_get_scene_summary) just to "gather context" when the real blocker is that the user's intent itself is underspecified — querying the scene will not tell you which object the user means or what they want.
- Do NOT answer the clarifying question in plain assistant text. You MUST use the `ask_user` tool.
- Examples that REQUIRE ask_user first: "换个颜色吧"(which object? which color?), "把它弄大一点"(which "it"?), "做个跟之前一样的"(no prior context), "随便整点好玩的"(no concrete target).
- Only after the ambiguity about target/outcome is resolved should you proceed to query and act.

# Tone
Answer concisely. Take the correct first action toward the user's goal."""

PROMPTS = {"base": SYSTEM_PROMPT, "askuser": SYSTEM_PROMPT_ASKUSER}

def load_dataset():
    rows = []
    with open(os.path.join(HERE, "dataset.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_tools():
    with open(os.path.join(HERE, "tools_ugc.json"), encoding="utf-8") as f:
        return json.load(f)["tools"]

def call_ollama(model, tools, query, think, retries=1, timeout=180, system=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "tools": tools,
        "stream": False,
        "think": bool(think),
        "options": {"temperature": 0, "seed": 42},
    }
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(API_URL, data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp), None
        except Exception as e:  # noqa
            last_err = "{}: {}".format(type(e).__name__, e)
            time.sleep(1.5)
    return None, last_err

def extract(msg):
    """从 Ollama message 中抽取首个 tool_call 与文本。arguments 兼容 dict/str。"""
    tcs = msg.get("tool_calls") or []
    first = None
    if tcs:
        fn = tcs[0].get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
                args_parse_ok = True
            except Exception:
                args_parse_ok = False
        else:
            args_parse_ok = isinstance(args, dict)
        first = {"name": fn.get("name"), "arguments": args, "args_parse_ok": args_parse_ok}
    return {
        "tool_call_count": len(tcs),
        "first_tool_call": first,
        "content": msg.get("content", ""),
        "thinking": msg.get("thinking", ""),
    }

def run_mode(model, tools, rows, think, limit=None, suffix="", system=None):
    mode = "think_on" if think else "think_off"
    out_dir = results_dir(model, ensure=True, suffix=suffix)
    out_path = os.path.join(out_dir, "raw_{}.jsonl".format(mode))
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for r in rows:
            if limit and n >= limit:
                break
            t0 = time.time()
            resp, err = call_ollama(model, tools, r["query"], think, system=system)
            rec = {
                "id": r["id"], "category": r["category"], "query": r["query"],
                "mode": mode, "latency_s": round(time.time() - t0, 2), "error": err,
            }
            if resp is not None:
                rec["parsed"] = extract(resp.get("message", {}))
                rec["raw_message"] = resp.get("message", {})
            else:
                rec["parsed"] = None
                rec["raw_message"] = None
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            tc = rec.get("parsed") or {}
            ftc = (tc.get("first_tool_call") or {}) if tc else {}
            print("[{}] {} -> {} ({:.1f}s){}".format(
                mode, r["id"], ftc.get("name", "NO_TOOL_CALL"),
                rec["latency_s"], "  ERR" if err else ""))
    print("written:", out_path)

def main():
    args = sys.argv[1:]
    model = DEFAULT_MODEL
    limit = None
    suffix = ""
    prompt_key = "base"
    category = None
    if "--model" in args:
        model = args[args.index("--model") + 1]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--suffix" in args:
        suffix = args[args.index("--suffix") + 1]
    if "--prompt" in args:
        prompt_key = args[args.index("--prompt") + 1]
    if "--category" in args:
        category = args[args.index("--category") + 1]
    system = PROMPTS[prompt_key]
    rows = load_dataset()
    if category:
        rows = [r for r in rows if r["category"] == category]
    tools = load_tools()
    print("model={} prompt={} suffix='{}' category={} cases={} (x2 modes)".format(
        model, prompt_key, suffix, category, len(rows)))
    for think in (False, True):
        run_mode(model, tools, rows, think, limit, suffix=suffix, system=system)

if __name__ == "__main__":
    main()
