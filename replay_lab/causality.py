"""Event-time ordering, causal graph validation, and gap detection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .canonical import deterministic_id, format_time, from_micros, micros, pairwise
from .errors import CausalityError
from .models import Event, Gap, event_index


REQUEST_TO_TERMINAL = {
    "tool_call_requested": {"tool_call_started", "tool_call_completed", "tool_call_failed"},
    "tool_call_started": {"tool_call_completed", "tool_call_failed"},
    "permission_requested": {"permission_granted", "permission_denied"},
    "approval_requested": {"approval_granted", "approval_denied"},
    "task_received": {"task_completed", "task_failed"},
}


@dataclass(frozen=True)
class CausalReport:
    ordered_events: tuple[Event, ...]
    errors: tuple[dict, ...]
    warnings: tuple[dict, ...]
    gaps: tuple[Gap, ...]
    roots: tuple[str, ...]
    leaves: tuple[str, ...]
    topological_ids: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "ordered_event_ids": [event.event_id for event in self.ordered_events],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "gaps": [gap.to_dict() for gap in self.gaps],
            "roots": list(self.roots),
            "leaves": list(self.leaves),
            "topological_ids": list(self.topological_ids),
        }


class CausalityAnalyzer:
    def __init__(self, *, clock_skew_ms: int = 2000, gap_threshold_ms: int = 10000) -> None:
        self.clock_skew_us = clock_skew_ms * 1000
        self.gap_threshold_us = gap_threshold_ms * 1000

    def analyze(self, events: Iterable[Event], *, strict: bool = False) -> CausalReport:
        values = list(events)
        index = event_index(values)
        errors: list[dict] = []
        warnings: list[dict] = []
        parents: dict[str, set[str]] = {event.event_id: set(event.parent_ids) for event in values}
        children: dict[str, set[str]] = defaultdict(set)
        for event in values:
            event_us = micros(event.event_time)
            observed_us = micros(event.observed_time)
            if observed_us + self.clock_skew_us < event_us:
                warnings.append({"type": "observed_before_event", "event_id": event.event_id,
                                 "difference_ms": (event_us - observed_us) / 1000})
            if event.event_id in event.parent_ids:
                errors.append({"type": "self_parent", "event_id": event.event_id})
            for parent_id in event.parent_ids:
                parent = index.get(parent_id)
                if not parent:
                    warnings.append({"type": "missing_parent", "event_id": event.event_id, "parent_id": parent_id})
                    continue
                children[parent_id].add(event.event_id)
                if micros(parent.event_time) > event_us + self.clock_skew_us:
                    errors.append({"type": "parent_after_child", "event_id": event.event_id,
                                   "parent_id": parent_id, "parent_time": parent.event_time,
                                   "child_time": event.event_time})
                if parent.incident_id != event.incident_id:
                    errors.append({"type": "cross_incident_parent", "event_id": event.event_id, "parent_id": parent_id})
        topological_ids, cycles = self._topological(values, parents, children)
        errors.extend({"type": "causal_cycle", "event_ids": cycle} for cycle in cycles)
        self._validate_correlations(values, warnings, errors)
        ordered = self._stable_event_order(values, topological_ids)
        gaps = self._detect_gaps(ordered, warnings)
        roots = tuple(sorted(event.event_id for event in values if not parents[event.event_id] or not any(p in index for p in parents[event.event_id])))
        leaves = tuple(sorted(event.event_id for event in values if not children[event.event_id]))
        report = CausalReport(tuple(ordered), tuple(errors), tuple(warnings), tuple(gaps), roots, leaves, tuple(topological_ids))
        if strict and not report.valid:
            raise CausalityError("causal validation failed", details=report.to_dict())
        return report

    def _topological(self, events: list[Event], parents: dict[str, set[str]], children: dict[str, set[str]]) -> tuple[list[str], list[list[str]]]:
        ids = {event.event_id for event in events}
        indegree = {event.event_id: sum(1 for parent in parents[event.event_id] if parent in ids) for event in events}
        event_map = {event.event_id: event for event in events}
        ready = sorted((event_id for event_id, degree in indegree.items() if degree == 0),
                       key=lambda item: self._event_key(event_map[item]))
        result: list[str] = []
        while ready:
            current = ready.pop(0)
            result.append(current)
            for child in sorted(children[current], key=lambda item: self._event_key(event_map[item])):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort(key=lambda item: self._event_key(event_map[item]))
        remaining = sorted(event_id for event_id, degree in indegree.items() if degree > 0)
        cycles: list[list[str]] = []
        while remaining:
            start = remaining[0]
            cycle = self._walk_cycle(start, parents, set(remaining))
            cycles.append(cycle)
            for item in cycle:
                if item in remaining:
                    remaining.remove(item)
        result.extend(sorted((item for item in indegree if item not in result), key=lambda item: self._event_key(event_map[item])))
        return result, cycles

    def _walk_cycle(self, start: str, parents: dict[str, set[str]], candidates: set[str]) -> list[str]:
        path: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while True:
            if current in positions:
                return path[positions[current]:] + [current]
            positions[current] = len(path)
            path.append(current)
            next_values = sorted(parent for parent in parents[current] if parent in candidates)
            if not next_values:
                return path
            current = next_values[0]

    def _event_key(self, event: Event) -> tuple:
        return (micros(event.event_time), micros(event.observed_time), event.source_id, event.sequence, event.event_id)

    def _stable_event_order(self, values: list[Event], topological_ids: list[str]) -> list[Event]:
        rank = {event_id: index for index, event_id in enumerate(topological_ids)}
        return sorted(values, key=lambda event: (micros(event.event_time), rank[event.event_id], micros(event.observed_time),
                                                  event.source_id, event.sequence, event.event_id))

    def _validate_correlations(self, events: list[Event], warnings: list[dict], errors: list[dict]) -> None:
        groups: dict[str, list[Event]] = defaultdict(list)
        for event in events:
            if event.correlation_id:
                groups[event.correlation_id].append(event)
        for correlation_id, values in sorted(groups.items()):
            ordered = sorted(values, key=self._event_key)
            for earlier, later in pairwise(ordered):
                allowed = REQUEST_TO_TERMINAL.get(earlier.kind)
                if allowed and later.kind not in allowed and later.kind not in {"policy_applied", "rate_limit_hit"}:
                    warnings.append({"type": "unexpected_correlation_transition", "correlation_id": correlation_id,
                                     "from": earlier.kind, "to": later.kind,
                                     "event_ids": [earlier.event_id, later.event_id]})
            terminals = [event for event in values if event.is_terminal]
            if len(terminals) > 1:
                outcomes = {(event.kind, event.outcome) for event in terminals}
                if len(outcomes) > 1:
                    errors.append({"type": "conflicting_terminal_outcomes", "correlation_id": correlation_id,
                                   "event_ids": [event.event_id for event in terminals]})

    def _detect_gaps(self, ordered: list[Event], warnings: list[dict]) -> list[Gap]:
        groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
        for event in ordered:
            groups[(event.incident_id, event.agent_id)].append(event)
        gaps: list[Gap] = []
        for (incident_id, agent_id), values in sorted(groups.items()):
            sequence_order = sorted(values, key=lambda event: (event.sequence, micros(event.event_time), event.event_id))
            for before, after in pairwise(sequence_order):
                missing = tuple(range(before.sequence + 1, after.sequence))
                elapsed = micros(after.event_time) - micros(before.event_time)
                reasons: list[str] = []
                if missing:
                    reasons.append("sequence_discontinuity")
                if elapsed > self.gap_threshold_us:
                    reasons.append("event_time_silence")
                if after.parent_ids and before.event_id not in after.parent_ids and any(parent not in {item.event_id for item in values} for parent in after.parent_ids):
                    reasons.append("unresolved_parent")
                if reasons:
                    gap_value = {
                        "incident_id": incident_id, "agent_id": agent_id,
                        "before": before.event_id, "after": after.event_id,
                        "missing": list(missing), "reasons": reasons,
                    }
                    gaps.append(Gap(
                        gap_id=deterministic_id("gap", gap_value), incident_id=incident_id, agent_id=agent_id,
                        start_time=before.event_time, end_time=after.event_time,
                        before_event_id=before.event_id, after_event_id=after.event_id,
                        missing_sequences=missing, reasons=tuple(reasons),
                    ))
        for warning in warnings:
            if warning["type"] == "missing_parent":
                child = next((event for event in ordered if event.event_id == warning["event_id"]), None)
                if child:
                    value = {"child": child.event_id, "missing_parent": warning["parent_id"]}
                    gaps.append(Gap(
                        deterministic_id("gap", value), child.incident_id, child.agent_id,
                        child.event_time, child.event_time, warning["parent_id"], child.event_id, (), ("missing_parent",),
                    ))
        unique = {gap.gap_id: gap for gap in gaps}
        return [unique[key] for key in sorted(unique)]

    def critical_path(self, report: CausalReport) -> list[str]:
        index = {event.event_id: event for event in report.ordered_events}
        parents = {event.event_id: [parent for parent in event.parent_ids if parent in index] for event in report.ordered_events}
        best: dict[str, tuple[int, list[str]]] = {}
        for event_id in report.topological_ids:
            event = index[event_id]
            weight = max(1, event.duration_ms)
            options = [best[parent] for parent in parents[event_id] if parent in best]
            if options:
                previous = max(options, key=lambda item: (item[0], item[1]))
                best[event_id] = (previous[0] + weight, previous[1] + [event_id])
            else:
                best[event_id] = (weight, [event_id])
        return max(best.values(), key=lambda item: (item[0], item[1]))[1] if best else []

    def event_time_windows(self, report: CausalReport, width_ms: int) -> list[dict]:
        if width_ms <= 0:
            raise CausalityError("window width must be positive")
        if not report.ordered_events:
            return []
        width_us = width_ms * 1000
        origin = micros(report.ordered_events[0].event_time)
        buckets: dict[int, list[Event]] = defaultdict(list)
        for event in report.ordered_events:
            buckets[(micros(event.event_time) - origin) // width_us].append(event)
        result: list[dict] = []
        for number, values in sorted(buckets.items()):
            start = origin + number * width_us
            result.append({
                "window": number, "start_time": from_micros(start), "end_time": from_micros(start + width_us),
                "event_ids": [event.event_id for event in values], "count": len(values),
                "loss_proxy": sum(event.loss_proxy for event in values),
                "applications": sorted({event.application for event in values if event.application}),
            })
        return result
