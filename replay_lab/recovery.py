"""Recovery planning and checkpoint equivalence verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import deterministic_id, digest_json, micros
from .errors import CheckpointError, ReplayError
from .models import Checkpoint, Event, PolicySpec
from .policies import PolicySandbox
from .replay import ReplayEngine, ReplayResult


@dataclass(frozen=True)
class RecoverySegment:
    segment_id: str
    checkpoint_id: str | None
    start_time: str | None
    end_time: str | None
    event_ids: tuple[str, ...]
    expected_stream_hash: str

    def to_dict(self) -> dict:
        return {"segment_id": self.segment_id, "checkpoint_id": self.checkpoint_id,
                "start_time": self.start_time, "end_time": self.end_time,
                "event_ids": list(self.event_ids), "expected_stream_hash": self.expected_stream_hash}


@dataclass(frozen=True)
class RecoveryVerification:
    equivalent: bool
    direct_state_hash: str
    resumed_state_hash: str
    direct_metrics: dict
    resumed_metrics: dict
    differences: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {"equivalent": self.equivalent, "direct_state_hash": self.direct_state_hash,
                "resumed_state_hash": self.resumed_state_hash, "direct_metrics": self.direct_metrics,
                "resumed_metrics": self.resumed_metrics, "differences": list(self.differences)}


class RecoveryPlanner:
    def __init__(self, engine: ReplayEngine | None = None) -> None:
        self.engine = engine or ReplayEngine()

    def segment(self, events: Iterable[Event], checkpoints: Iterable[Checkpoint]) -> list[RecoverySegment]:
        values = sorted(events, key=lambda event: (micros(event.event_time), event.sequence, event.event_id))
        points = sorted(checkpoints, key=lambda item: (micros(item.event_time), item.checkpoint_id))
        self._validate_points(points)
        segments: list[RecoverySegment] = []
        previous_time: int | None = None
        previous_checkpoint: str | None = None
        for point in points + [None]:
            end_time = micros(point.event_time) if point else None
            selected = [event for event in values
                        if (previous_time is None or micros(event.event_time) > previous_time)
                        and (end_time is None or micros(event.event_time) <= end_time)]
            if selected or point is None:
                identity = {"checkpoint_id": previous_checkpoint,
                            "events": [event.evidence_hash for event in selected],
                            "start": previous_time, "end": end_time}
                segments.append(RecoverySegment(
                    deterministic_id("segment", identity), previous_checkpoint,
                    selected[0].event_time if selected else None,
                    selected[-1].event_time if selected else None,
                    tuple(event.event_id for event in selected), digest_json(identity["events"]),
                ))
            if point:
                previous_time = end_time
                previous_checkpoint = point.checkpoint_id
        return segments

    def verify_resume(self, events: Iterable[Event], checkpoint: Checkpoint,
                      policies: Iterable[PolicySpec] = (), scenario_id: str = "recovery") -> RecoveryVerification:
        values = list(events)
        sandbox = PolicySandbox(tuple(policies))
        direct = self.engine.replay(values, scenario_id=scenario_id, policies=sandbox)
        resumed = self.engine.replay(values, scenario_id=scenario_id, policies=PolicySandbox(tuple(policies)),
                                     checkpoint=checkpoint)
        differences = self._diff_states(direct, resumed)
        return RecoveryVerification(not differences, direct.state_hash, resumed.state_hash,
                                    direct.metrics.to_dict(), resumed.metrics.to_dict(), tuple(differences))

    def create_verified_checkpoint(self, prefix_events: Iterable[Event], checkpoint_id: str,
                                   event_time: str, policies: Iterable[PolicySpec] = (),
                                   scenario_id: str = "checkpoint") -> Checkpoint:
        values = list(prefix_events)
        if not values:
            raise CheckpointError("checkpoint requires at least one prefix event")
        result = self.engine.replay(values, scenario_id=scenario_id, policies=PolicySandbox(tuple(policies)))
        checkpoint = self.engine.checkpoint(result, checkpoint_id, event_time)
        self.engine.verify_checkpoint(checkpoint)
        return checkpoint

    def select_resume_point(self, checkpoints: Iterable[Checkpoint], target_time: str) -> Checkpoint | None:
        target = micros(target_time)
        eligible = [checkpoint for checkpoint in checkpoints if micros(checkpoint.event_time) <= target]
        if not eligible:
            return None
        return max(eligible, key=lambda item: (micros(item.event_time), item.checkpoint_id))

    def _validate_points(self, checkpoints: list[Checkpoint]) -> None:
        seen: set[str] = set()
        previous: int | None = None
        incident_id: str | None = None
        for checkpoint in checkpoints:
            self.engine.verify_checkpoint(checkpoint)
            if checkpoint.checkpoint_id in seen:
                raise CheckpointError("duplicate checkpoint identifier", details={"checkpoint_id": checkpoint.checkpoint_id})
            seen.add(checkpoint.checkpoint_id)
            if incident_id is None:
                incident_id = checkpoint.incident_id
            elif checkpoint.incident_id != incident_id:
                raise CheckpointError("checkpoints span multiple incidents")
            current = micros(checkpoint.event_time)
            if previous is not None and current < previous:
                raise CheckpointError("checkpoints are not monotonic")
            previous = current

    def _diff_states(self, direct: ReplayResult, resumed: ReplayResult) -> list[dict]:
        left = direct.state.to_dict()
        right = resumed.state.to_dict()
        differences: list[dict] = []
        ignored = {"timeline"}
        for key in sorted(set(left) | set(right)):
            if key in ignored:
                continue
            if left.get(key) != right.get(key):
                differences.append({"field": key, "direct": left.get(key), "resumed": right.get(key)})
        return differences

    def repair_checkpoint_metadata(self, checkpoint: Checkpoint, observed_events: Iterable[Event]) -> dict:
        """Diagnose checkpoint anchors without changing its signed state payload."""
        values = [event for event in observed_events if micros(event.event_time) <= micros(checkpoint.event_time)]
        known = {event.event_id for event in values}
        missing = [event_id for event_id in checkpoint.last_event_ids if event_id not in known]
        latest = sorted(values, key=lambda event: (micros(event.event_time), event.event_id))[-3:]
        return {"checkpoint_id": checkpoint.checkpoint_id, "state_hash_valid": digest_json(checkpoint.state) == checkpoint.state_hash,
                "missing_anchor_ids": missing, "suggested_anchor_ids": [event.event_id for event in latest],
                "safe_to_replace_metadata": not missing and bool(latest)}
