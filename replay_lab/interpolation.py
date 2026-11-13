"""Confidence-scored candidate interpolation that never mutates evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from .canonical import clamp, deterministic_id, from_micros, micros
from .models import Candidate, Event, Gap


TRANSITIONS: dict[tuple[str, str], tuple[str, ...]] = {
    ("tool_call_requested", "tool_call_completed"): ("tool_call_started",),
    ("tool_call_requested", "tool_call_failed"): ("tool_call_started",),
    ("permission_requested", "tool_call_started"): ("permission_granted",),
    ("approval_requested", "tool_call_started"): ("approval_granted",),
    ("task_received", "task_completed"): ("tool_call_requested", "tool_call_started", "tool_call_completed"),
    ("task_received", "task_failed"): ("tool_call_requested", "tool_call_started", "tool_call_failed"),
    ("agent_started", "task_completed"): ("task_received",),
}


class Interpolator:
    def __init__(self, *, max_candidates: int = 4, confidence_floor: float = 0.15) -> None:
        self.max_candidates = max_candidates
        self.confidence_floor = confidence_floor

    def generate(self, gaps: Iterable[Gap], events: Iterable[Event]) -> list[Candidate]:
        values = list(events)
        by_id = {event.event_id: event for event in values}
        by_agent: dict[str, list[Event]] = defaultdict(list)
        by_correlation: dict[str, list[Event]] = defaultdict(list)
        for event in values:
            by_agent[event.agent_id].append(event)
            if event.correlation_id:
                by_correlation[event.correlation_id].append(event)
        for group in by_agent.values():
            group.sort(key=lambda item: (micros(item.event_time), item.sequence, item.event_id))
        candidates: list[Candidate] = []
        for gap in gaps:
            before = by_id.get(gap.before_event_id or "")
            after = by_id.get(gap.after_event_id or "")
            proposed: list[Candidate] = []
            proposed.extend(self._transition_candidates(gap, before, after, by_correlation))
            proposed.extend(self._sequence_candidates(gap, before, after))
            proposed.extend(self._parent_candidates(gap, before, after))
            proposed.extend(self._silence_candidates(gap, before, after, by_agent.get(gap.agent_id, [])))
            unique: dict[str, Candidate] = {}
            for candidate in proposed:
                if candidate.confidence < self.confidence_floor:
                    continue
                key = "|".join(event.kind for event in candidate.events) + ":" + "|".join(event.correlation_id or "" for event in candidate.events)
                existing = unique.get(key)
                if not existing or candidate.confidence > existing.confidence:
                    unique[key] = candidate
            ordered = sorted(unique.values(), key=lambda item: (-item.confidence, item.candidate_id))[: self.max_candidates]
            candidates.extend(ordered)
        return candidates

    def _transition_candidates(self, gap: Gap, before: Event | None, after: Event | None,
                               correlations: dict[str, list[Event]]) -> list[Candidate]:
        if not before or not after:
            return []
        kinds = TRANSITIONS.get((before.kind, after.kind), ())
        if not kinds:
            return []
        correlation = before.correlation_id or after.correlation_id
        confidence = 0.82
        rationale = ["known_protocol_transition", "bounded_by_observed_events"]
        contradictions: list[str] = []
        if before.correlation_id and after.correlation_id and before.correlation_id != after.correlation_id:
            confidence -= 0.3
            contradictions.append("correlation_mismatch")
        if correlation and any(event.kind in kinds for event in correlations.get(correlation, [])):
            confidence -= 0.4
            contradictions.append("similar_event_already_observed")
        events = self._synthesize(gap, before, after, kinds, correlation, confidence, "protocol")
        candidate = self._candidate(gap, events, confidence, rationale, contradictions)
        return [candidate]

    def _sequence_candidates(self, gap: Gap, before: Event | None, after: Event | None) -> list[Candidate]:
        if not before or not after or not gap.missing_sequences:
            return []
        options: list[Candidate] = []
        if before.kind == "tool_call_requested":
            kinds = tuple("tool_call_started" for _ in gap.missing_sequences[:1])
        elif after.kind in {"resource_written", "resource_deleted", "message_sent"}:
            kinds = ("tool_call_started",) if len(gap.missing_sequences) == 1 else ("permission_granted", "tool_call_started")
        elif after.kind in {"task_completed", "task_failed"}:
            kinds = ("tool_call_completed",) if after.kind == "task_completed" else ("tool_call_failed",)
        else:
            kinds = ()
        if kinds:
            confidence = clamp(0.45 + 0.08 * min(len(gap.missing_sequences), 3), 0.0, 0.72)
            contradictions: list[str] = []
            if before.correlation_id and after.correlation_id and before.correlation_id != after.correlation_id:
                confidence = max(0.1, confidence - 0.18)
                contradictions.append("correlation_mismatch")
            events = self._synthesize(gap, before, after, kinds, before.correlation_id or after.correlation_id,
                                      confidence, "sequence")
            options.append(self._candidate(gap, events, confidence,
                                           ["sequence_numbers_missing", "local_state_machine_fit"], contradictions))
        return options

    def _parent_candidates(self, gap: Gap, before: Event | None, after: Event | None) -> list[Candidate]:
        if "missing_parent" not in gap.reasons or not after or not gap.before_event_id:
            return []
        kind = "tool_call_started" if after.kind in {"tool_call_completed", "tool_call_failed", "resource_written", "resource_deleted"} else "task_received"
        confidence = 0.38
        pseudo_before = before or replace(after, event_id="boundary", event_time=after.event_time,
                                           observed_time=after.observed_time, sequence=max(0, after.sequence - 2))
        events = self._synthesize(gap, pseudo_before, after, (kind,), after.correlation_id, confidence, "parent")
        events = tuple(replace(event, parent_ids=(gap.before_event_id,)) for event in events)
        return [self._candidate(gap, events, confidence, ["child_references_missing_parent", "kind_inferred_from_child"],
                                ["missing_parent_payload_unavailable"])]

    def _silence_candidates(self, gap: Gap, before: Event | None, after: Event | None, agent_events: list[Event]) -> list[Candidate]:
        if "event_time_silence" not in gap.reasons or not before or not after:
            return []
        elapsed_ms = (micros(after.event_time) - micros(before.event_time)) / 1000
        failures = sum(1 for event in agent_events if event.outcome in {"failed", "denied", "timeout"})
        if failures:
            kinds = ("rate_limit_hit",) if any(event.kind == "rate_limit_hit" for event in agent_events) else ("tool_call_failed",)
            confidence = clamp(0.25 + min(failures, 4) * 0.06 + min(elapsed_ms / 120000, 0.12), 0.0, 0.55)
            events = self._synthesize(gap, before, after, kinds, before.correlation_id or after.correlation_id,
                                      confidence, "silence_failure")
            return [self._candidate(gap, events, confidence, ["long_silence", "agent_failure_history"],
                                    ["no_direct_evidence_for_failure_kind"])]
        return []

    def _synthesize(self, gap: Gap, before: Event, after: Event, kinds: tuple[str, ...], correlation: str | None,
                    confidence: float, source: str) -> tuple[Event, ...]:
        start = micros(before.event_time)
        end = micros(after.event_time)
        span = max(len(kinds) + 1, 2)
        result: list[Event] = []
        previous_id = before.event_id
        sequences = list(gap.missing_sequences)
        for index, kind in enumerate(kinds, 1):
            event_us = start + ((end - start) * index // span)
            sequence = sequences[index - 1] if index - 1 < len(sequences) else before.sequence + index
            identity = {"gap": gap.gap_id, "kind": kind, "index": index, "source": source,
                        "event_time": from_micros(event_us), "sequence": sequence}
            event_id = deterministic_id("hyp", identity)
            event = Event.from_dict({
                "event_id": event_id, "incident_id": gap.incident_id, "source_id": f"interpolation:{source}",
                "agent_id": gap.agent_id, "kind": kind, "event_time": from_micros(event_us),
                "observed_time": after.observed_time, "sequence": sequence, "correlation_id": correlation,
                "parent_ids": [previous_id] if previous_id else [], "permissions": list(before.permissions),
                "application": before.application or after.application, "tool": before.tool or after.tool,
                "resource": before.resource or after.resource, "outcome": "hypothetical",
                "duration_ms": 0, "loss_proxy": 0.0, "availability_value": 0.0,
                "payload": {"interpolation_gap_id": gap.gap_id, "method": source},
                "tags": ["hypothesis", "non_evidence"], "synthetic": True, "confidence": confidence,
            })
            result.append(event)
            previous_id = event_id
        return tuple(result)

    def _candidate(self, gap: Gap, events: tuple[Event, ...], confidence: float,
                   rationale: list[str], contradictions: list[str]) -> Candidate:
        identity = {"gap_id": gap.gap_id, "events": [event.to_dict() for event in events],
                    "confidence": round(confidence, 6), "rationale": rationale, "contradictions": contradictions}
        return Candidate(deterministic_id("candidate", identity), gap.gap_id, events,
                         round(clamp(confidence, 0.0, 1.0), 6), tuple(rationale), tuple(contradictions))

    def overlay(self, evidence: Iterable[Event], candidate: Candidate | None) -> list[Event]:
        """Return a new replay view; original evidence objects and journal remain untouched."""
        values = list(evidence)
        if candidate:
            values.extend(candidate.events)
        return sorted(values, key=lambda event: (micros(event.event_time), event.synthetic, event.sequence, event.event_id))
