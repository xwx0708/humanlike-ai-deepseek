"""真实调用本地 Qwen 跑一遍整理流程。默认不运行，用 `pytest -m live` 手动运行（约 1～2 分钟）。"""

import pytest

from llm import subconscious_model
from memory.store import MemoryStore
from memory.subconscious import consolidate

pytestmark = pytest.mark.live
DAY = 86400.0


def test_consolidate_with_real_qwen():
    store, qwen = MemoryStore(":memory:"), subconscious_model()
    c0 = store.new_conversation(0)
    store.add_turn(c0.id, "user", "今天好累，我妈最近老咳嗽，晚上睡不好", 0)
    store.add_turn(c0.id, "assistant", "让阿姨去医院看看吧", 30)
    consolidate(store, qwen, c0.id, 600)
    [old] = store.segments()

    c = store.new_conversation(180 * DAY)
    for i, (role, text) in enumerate(
        [
            ("user", "我妈住院了，之前那个咳嗽查出来是肺炎"),
            ("assistant", "现在情况怎么样？"),
            ("user", "医生说要住一周，母亲精神还行"),
            ("assistant", "你也注意休息"),
            ("user", "对了你记住这个：我对花生过敏"),
            ("assistant", "好的，记住了"),
        ]
    ):
        store.add_turn(c.id, role, text, 180 * DAY + i * 60)
    report = consolidate(store, qwen, c.id, 180 * DAY + 3600)

    assert len(report.segment_ids) >= 2  # 住院和过敏是两件事
    assert store.resolve("我妈") == store.resolve("母亲")  # 别名表认出是同一个人
    hospital = next(s for s in report.segment_ids if old.person_ids[0] in store.get_segment(s).person_ids)
    assert set(store.threads_of_segment(hospital)) & set(store.threads_of_segment(old.id))  # 挂到同一条事件线
    allergy = max(report.segment_ids, key=lambda s: store.get_segment(s).emotion.importance)
    assert store.get_segment(allergy).emotion.importance >= 0.9  # "记住这个"
