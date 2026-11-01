"""门面的显式方法转发，签名沿用拆分前的 MemoryStore。"""

from typing import Any

from contracts.memory import Person, ProfileFact, Segment, Thread
from memory.forgetting import ForgettingKnobs
from memory.tuning import GlobalChange

from .profile import ProfileStore
from .segments import SegmentStore
from .threads import ThreadStore
from .tuning import TuningStore


class _StoreMethods:
    _segments: SegmentStore
    _threads: ThreadStore
    _profile: ProfileStore
    _tuning: TuningStore

    def forgotten_turn_ids(self) -> set[int]:
        return self._threads.forgotten_turn_ids()

    def add_boundary(self, turn_id: int, reason: str) -> None:
        return self._segments.add_boundary(turn_id, reason)

    def boundaries(self, conversation_id: int) -> dict[int, list[str]]:
        return self._segments.boundaries(conversation_id)

    def add_segment(self, seg: Segment, *, job: dict[str, Any] | None = None) -> Segment:
        return self._segments.add_segment(seg, job=job)

    def save_consolidation_job(self, sid: int, batch_id: int, stage: str, payload: dict) -> None:
        return self._segments.save_consolidation_job(sid, batch_id, stage, payload)

    def consolidation_job(self, sid: int) -> tuple[int, str, dict]:
        return self._segments.consolidation_job(sid)

    def pending_consolidations(self, cid: int) -> list[int]:
        return self._segments.pending_consolidations(cid)

    def consolidation_batch(self, batch_id: int) -> set[int]:
        return self._segments.consolidation_batch(batch_id)

    def adjusted_in_batch(self, batch_id: int) -> set[str]:
        return self._segments.adjusted_in_batch(batch_id)

    def get_segment(self, sid: int) -> Segment:
        return self._segments.get_segment(sid)

    def save_segment_state(self, seg: Segment) -> None:
        return self._segments.save_segment_state(seg)

    def segments(self, conversation_id: int | None = None, state: str | None = None) -> list[Segment]:
        return self._segments.segments(conversation_id, state)

    def link_segments(self, from_id: int, to_id: int, kind: str = "next") -> None:
        return self._segments.link_segments(from_id, to_id, kind)

    def segment_links(self, conversation_id: int) -> list[tuple[int, int, str]]:
        return self._segments.segment_links(conversation_id)

    def new_thread(self, title: str, summary: str, ts: float) -> Thread:
        return self._threads.new_thread(title, summary, ts)

    def attach(self, thread_id: int, segment_id: int, confidence: float, ts: float) -> None:
        return self._threads.attach(thread_id, segment_id, confidence, ts)

    def get_thread(self, tid: int) -> Thread:
        return self._threads.get_thread(tid)

    def update_thread(
        self, tid: int, *, title: str | None = None, summary: str | None = None, ongoing: bool | None = None
    ) -> None:
        return self._threads.update_thread(tid, title=title, summary=summary, ongoing=ongoing)

    def thread_stats(self, tid: int) -> dict[str, Any]:
        return self._threads.thread_stats(tid)

    def thread_has_profile_fact(self, tid: int) -> bool:
        return self._threads.thread_has_profile_fact(tid)

    def resident_threads(self) -> list[int]:
        return self._threads.resident_threads()

    def set_resident(
        self, tid: int, ts: float, resident: bool, reason: str, score: float | None = None
    ) -> None:
        return self._threads.set_resident(tid, ts, resident, reason, score)

    def exempt_marks(self) -> dict[int, str]:
        return self._threads.exempt_marks()

    def mark_exempt(self, tid: int, reason: str) -> None:
        return self._threads.mark_exempt(tid, reason)

    def release_exempt(self, tid: int, ts: float, reason: str) -> None:
        return self._threads.release_exempt(tid, ts, reason)

    def resident_log(self) -> list[dict[str, Any]]:
        return self._threads.resident_log()

    def threads_of_segment(self, segment_id: int) -> list[int]:
        return self._threads.threads_of_segment(segment_id)

    def relate_threads(self, a: int, b: int, note: str) -> None:
        return self._threads.relate_threads(a, b, note)

    def merge_threads(self, keep: int, drop: int, ts: float) -> None:
        return self._threads.merge_threads(keep, drop, ts)

    def _invalidate_summaries(self, sids: list[int]) -> None:
        return self._threads._invalidate_summaries(sids)

    def forget_thread_to_archive(self, tid: int, ts: float) -> list[int]:
        return self._threads.forget_thread_to_archive(tid, ts)

    def restore_thread(self, tid: int) -> None:
        return self._threads.restore_thread(tid)

    def forgotten_thread_ids(self) -> list[int]:
        return self._threads.forgotten_thread_ids()

    def _facts_only_from(self, sids: list[int]) -> list[int]:
        return self._threads._facts_only_from(sids)

    def erase_thread(self, tid: int, ts: float, reason: str) -> tuple[int, int]:
        return self._threads.erase_thread(tid, ts, reason)

    def erase_log(self) -> list[dict[str, Any]]:
        return self._threads.erase_log()

    def live_threads(self) -> list[int]:
        return self._threads.live_threads()

    def resolve(self, name: str) -> int | None:
        return self._profile.resolve(name)

    def upsert_person(self, name: str, aliases: list[str] = ()) -> int:
        return self._profile.upsert_person(name, aliases)

    def person(self, pid: int) -> Person:
        return self._profile.person(pid)

    def load_global_knobs(self) -> ForgettingKnobs:
        return self._tuning.load_global_knobs()

    def save_global_knobs(self, knobs: ForgettingKnobs) -> None:
        return self._tuning.save_global_knobs(knobs)

    def log_tuning(
        self, scope: str, target: str, change: GlobalChange, evidence: dict[str, Any] | None = None
    ) -> None:
        return self._tuning.log_tuning(scope, target, change, evidence)

    def tuning_log(self, scope: str | None = None) -> list[tuple[str, str, GlobalChange]]:
        return self._tuning.tuning_log(scope)

    def add_profile_fact(
        self, person_id: int | None, category: str, fact: str, ts: float, segment_id: int | None = None
    ) -> int:
        return self._profile.add_profile_fact(person_id, category, fact, ts, segment_id)

    def profile(
        self, person_id: int | None = None, *, everyone: bool = False, archived: bool = False
    ) -> list[ProfileFact]:
        return self._profile.profile(person_id, everyone=everyone, archived=archived)

    def fact_text(self, fact_id: int) -> str | None:
        return self._profile.fact_text(fact_id)

    def archive_profile_fact(self, old_id: int, new_id: int, ts: float) -> None:
        return self._profile.archive_profile_fact(old_id, new_id, ts)

    def log_profile_conflict(
        self,
        old_id: int,
        new_id: int,
        ts: float,
        firmness: float,
        probs: dict[str, float],
        roll: float,
        outcome: str,
        reason: str,
    ) -> None:
        return self._profile.log_profile_conflict(old_id, new_id, ts, firmness, probs, roll, outcome, reason)

    def profile_conflicts(self) -> list[dict[str, Any]]:
        return self._profile.profile_conflicts()

    def add_evocation(self, fact_id: int, trigger_fact_id: int | None, ts: float) -> None:
        return self._profile.add_evocation(fact_id, trigger_fact_id, ts)

    def pending_evocations(self) -> list[dict[str, Any]]:
        return self._profile.pending_evocations()

    def consume_evocation(self, evocation_id: int) -> None:
        return self._profile.consume_evocation(evocation_id)

    def threads_with_people(self, person_ids: list[int]) -> list[int]:
        return self._threads.threads_with_people(person_ids)

    def threads_mentioning(self, text: str, exclude_turns: list[int] = ()) -> list[int]:
        return self._threads.threads_mentioning(text, exclude_turns)

    def thread_text(self, tid: int) -> str:
        return self._threads.thread_text(tid)
