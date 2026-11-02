"""设置页中英文：英文文案 en.json 要覆盖注册表里的每一项，页面才能整页切换；缺了回落到中文。"""

import json
from pathlib import Path

from settings_page.schema import schema
from settings_page.server import EN


def test_english_covers_every_registered_item():
    en = json.loads(Path(EN).read_text(encoding="utf-8"))
    s = schema()
    missing = []
    for sec in s["sections"]:
        if sec["id"] not in en["sections"]:
            missing.append(sec["id"])
        for f in sec["fields"]:
            key = f"{sec['id']}.{f['key']}"
            e = en["fields"].get(key)
            if not e or not e.get("label") or not e.get("help"):
                missing.append(key)
            elif f.get("option_labels") and len(e.get("options", [])) != len(f["option_labels"]):
                missing.append(key + ".options")
            if f.get("unit") and f["unit"] not in en["units"]:
                missing.append(key + ".unit:" + f["unit"])
    missing += [f["key"] for f in s["features"] if f["key"] not in en["features"]]
    missing += [w["key"] for w in s["word_lists"] if w["key"] not in en["word_lists"]]
    assert not missing, f"en.json 缺少：{missing}"
    stale = [
        k
        for k in en["fields"]
        if k not in {f"{s['id']}.{f['key']}" for s in s["sections"] for f in s["fields"]}
    ]
    assert not stale, f"en.json 里这些参数已不存在：{stale}"
