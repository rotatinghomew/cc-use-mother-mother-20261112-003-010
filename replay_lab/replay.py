"""Deterministic event replay, checkpoint restore, and state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import deterministic_id, digest_json, micros
from .errors import CheckpointError, ReplayError
from .models import Checkpoint, Decision, Event, ReplayState, RunMetrics
from .policies import PolicySandbox


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    scenario_id: str
    incident_id: str
    state: ReplayState
    metrics: RunMetrics
    decisions: tuple[Decision, ...]
    processed_event_ids: tuple[str, ...]
    skipped_event_ids: tuple[str, ...]
    start_checkpoint_id: str | None
    event_stream_hash: str
    state_hash: str

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "scenario_id": self.scenario_id, "incident_id": self.incident_id,
            "metrics": self.metrics.to_dict(), "decisions": [decision.to_dict() for decision in self.decisions],
            "processed_event_ids": list(self.processed_event_ids), "skipped_event_ids": list(self.skipped_event_ids),
            "start_checkpoint_id": self.start_checkpoint_id, "event_stream_hash": self.event_stream_hash,
            "state_hash": self.state_hash, "state": self.state.to_dict(),
        }


class ReplayEngine:
    def replay(self, events: Iterable[Event], *, scenario_id: str, policies: PolicySandbox | None = None,
               checkpoint: Checkpoint | None = None, stop_after: str | None = None,
               include_synthetic: bool = True) -> ReplayResult:
        values = list(events)
        if not values and not checkpoint:
            raise ReplayError("cannot replay empty event stream without checkpoint")
        incident_id = checkpoint.incident_id if checkpoint else values[0].incident_id
        if any(event.incident_id != incident_id for event in values):
            raise ReplayError("event stream contains multiple incidents")
        state, checkpoint_time = self._restore(checkpoint, incident_id)
        sandbox = policies or PolicySandbox(())
        runtime = sandbox.new_runtime()
        processed: list[str] = []
        skipped: list[str] = []
        selected = sorted(values, key=lambda event: (micros(event.event_time), event.sequence, event.source_id,
                                                       event.synthetic, event.event_id))
        for event in selected:
            if checkpoint_time is not None and micros(event.event_time) <= checkpoint_time:
                skipped.append(event.event_id)
                continue
            if event.synthetic and not include_synthetic:
                skipped.append(event.event_id)
                continue
            decision = sandbox.decide(event, state, runtime)
            self._apply_decision_costs(state, event, decision)
            if decision.allowed:
                effective = self._effective_event(event, decision)
                self._apply_event(state, effective)
                state.applied_events += 1
                processed.append(event.event_id)
            else:
                state.blocked_events += 1
                if event.loss_proxy > 0 or event.is_mutation:
                    state.contained_events += 1
                skipped.append(event.event_id)
            sandbox.observe(event, decision, runtime)
            state.timeline.append(self._timeline_entry(event, decision, state))
            if stop_after and event.event_id == stop_after:
                break
        metrics = self._metrics(state)
        stream_identity = [event.to_dict() for event in selected]
        state_hash = state.state_hash()
        identity = {"scenario_id": scenario_id, "incident_id": incident_id,
                    "checkpoint": checkpoint.checkpoint_id if checkpoint else None,
                    "event_stream_hash": digest_json(stream_identity), "state_hash": state_hash,
                    "decisions": [decision.to_dict() for decision in runtime.decisions]}
        run_id = deterministic_id("run", identity)
        return ReplayResult(run_id, scenario_id, incident_id, state, metrics, tuple(runtime.decisions),
                            tuple(processed), tuple(skipped), checkpoint.checkpoint_id if checkpoint else None,
                            digest_json(stream_identity), state_hash)

    def checkpoint(self, result: ReplayResult, checkpoint_id: str, event_time: str) -> Checkpoint:
        last_ids = result.processed_event_ids[-3:]
        state = result.state.to_dict()
        return Checkpoint(checkpoint_id, result.incident_id, event_time, tuple(last_ids), state, digest_json(state), "replay")

    def verify_checkpoint(self, checkpoint: Checkpoint) -> None:
        if digest_json(checkpoint.state) != checkpoint.state_hash:
            raise CheckpointError("checkpoint hash mismatch", details={"checkpoint_id": checkpoint.checkpoint_id})
        state = ReplayState.from_dict(checkpoint.state)
        if state.incident_id != checkpoint.incident_id:
            raise CheckpointError("checkpoint incident mismatch", details={"checkpoint_id": checkpoint.checkpoint_id})

    def _restore(self, checkpoint: Checkpoint | None, incident_id: str) -> tuple[ReplayState, int | None]:
        if not checkpoint:
            return ReplayState(incident_id=incident_id), None
        self.verify_checkpoint(checkpoint)
        return ReplayState.from_dict(checkpoint.state), micros(checkpoint.event_time)

    def _effective_event(self, event: Event, decision: Decision) -> Event:
        permissions = event.permissions
        if "effective_permissions" in decision.mutations:
            permissions = tuple(decision.mutations["effective_permissions"])
        if permissions == event.permissions:
            return event
        raw = event.to_dict()
        raw["permissions"] = list(permissions)
        raw.pop("evidence_hash", None)
        raw["payload"] = dict(event.payload)
        raw["payload"]["counterfactual_effective_permissions"] = list(permissions)
        raw["synthetic"] = True
        return Event.from_dict(raw)

    def _apply_decision_costs(self, state: ReplayState, event: Event, decision: Decision) -> None:
        state.total_delay_ms += decision.delay_ms
        if decision.allowed:
            state.total_loss += event.loss_proxy
        else:
            state.total_availability += event.availability_value
            state.total_availability += max(0.0, event.duration_ms / 1000.0 * 0.01)
        if decision.delay_ms:
            state.total_availability += decision.delay_ms / 1000.0 * max(event.availability_value, 0.01)

    def _apply_event(self, state: ReplayState, event: Event) -> None:
        state.counters[event.kind] = state.counters.get(event.kind, 0) + 1
        if event.kind == "agent_started":
            state.active_agents.add(event.agent_id)
        elif event.kind == "agent_stopped":
            state.active_agents.discard(event.agent_id)
        elif event.kind == "task_received":
            key = event.correlation_id or event.event_id
            state.tasks[key] = "received"
        elif event.kind in {"task_completed", "task_failed"}:
            key = event.correlation_id or event.event_id
            state.tasks[key] = "completed" if event.kind == "task_completed" else "failed"
        elif event.kind == "permission_requested":
            key = event.correlation_id or event.event_id
            state.pending_approvals[key] = {"type": "permission", "permissions": list(event.permissions),
                                             "agent_id": event.agent_id, "event_id": event.event_id}
        elif event.kind == "permission_granted":
            state.permissions.setdefault(event.agent_id, set()).update(event.permissions)
            if event.correlation_id:
                state.pending_approvals.pop(event.correlation_id, None)
        elif event.kind == "permission_denied":
            if event.correlation_id:
                state.pending_approvals.pop(event.correlation_id, None)
        elif event.kind == "approval_requested":
            key = event.correlation_id or event.event_id
            state.pending_approvals[key] = {"type": "approval", "agent_id": event.agent_id,
                                             "permissions": list(event.permissions), "event_id": event.event_id}
        elif event.kind in {"approval_granted", "approval_denied"}:
            if event.correlation_id:
                state.pending_approvals.pop(event.correlation_id, None)
        elif event.kind in {"tool_call_requested", "tool_call_started"}:
            key = event.correlation_id or event.event_id
            state.open_calls[key] = {"agent_id": event.agent_id, "tool": event.tool,
                                     "application": event.application, "status": event.kind,
                                     "event_id": event.event_id}
        elif event.kind in {"tool_call_completed", "tool_call_failed"}:
            key = event.correlation_id or event.event_id
            call = state.open_calls.pop(key, {"agent_id": event.agent_id, "tool": event.tool,
                                               "application": event.application})
            call["status"] = event.kind
            call["outcome"] = event.outcome
        elif event.kind in {"resource_read", "resource_written", "resource_deleted"}:
            resource = event.resource or f"unknown:{event.event_id}"
            record = state.resources.setdefault(resource, {"reads": 0, "writes": 0, "deletes": 0,
                                                             "agents": [], "last_event_id": None})
            if event.kind == "resource_read":
                record["reads"] += 1
            elif event.kind == "resource_written":
                record["writes"] += 1
            else:
                record["deletes"] += 1
                record["deleted"] = True
            if event.agent_id not in record["agents"]:
                record["agents"].append(event.agent_id)
                record["agents"].sort()
            record["last_event_id"] = event.event_id
            record["last_outcome"] = event.outcome
        if event.application:
            state.applications[event.application] = event.event_id

    def _timeline_entry(self, event: Event, decision: Decision, state: ReplayState) -> dict:
        return {
            "event_id": event.event_id, "event_time": event.event_time, "observed_time": event.observed_time,
            "agent_id": event.agent_id, "kind": event.kind, "application": event.application,
            "tool": event.tool, "resource": event.resource, "synthetic": event.synthetic,
            "confidence": event.confidence, "decision": decision.to_dict(),
            "cumulative_loss": round(state.total_loss, 6),
            "cumulative_availability_cost": round(state.total_availability, 6),
            "cumulative_delay_ms": state.total_delay_ms,
        }

    def _metrics(self, state: ReplayState) -> RunMetrics:
        impacted_agents = {item["agent_id"] for item in state.timeline if item["decision"]["allowed"] and
                           (item["kind"] in {"resource_written", "resource_deleted", "message_sent", "tool_call_started"}
                            or item["decision"]["action"] != "allow")}
        impacted_apps = {item["application"] for item in state.timeline if item["application"] and item["decision"]["allowed"]}
        impacted_resources = {item["resource"] for item in state.timeline if item["resource"] and item["decision"]["allowed"] and
                              item["kind"] in {"resource_written", "resource_deleted"}}
        return RunMetrics(len(impacted_agents), len(impacted_apps), len(impacted_resources),
                          round(state.total_loss, 6), round(state.total_availability, 6),
                          state.total_delay_ms, state.blocked_events, state.contained_events, state.applied_events)
