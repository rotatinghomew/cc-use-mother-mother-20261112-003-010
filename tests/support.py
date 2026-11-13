from __future__ import annotations

import json
import tempfile
from pathlib import Path

from replay_lab.canonical import digest_json
from replay_lab.models import Checkpoint, Event, PolicySpec, ReplayState


BASE_TIME = "2026-11-12T02:00:00.000000Z"


def timestamp(seconds: int, millis: int = 0) -> str:
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 11, 12, 2, 0, 0, tzinfo=timezone.utc)
    value = base + timedelta(seconds=seconds, milliseconds=millis)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def event_dict(event_id: str, sequence: int, kind: str, seconds: int, *,
               observed_seconds: int | None = None, source_id: str = "collector-a",
               agent_id: str = "agent-alpha", incident_id: str = "inc-001",
               correlation_id: str | None = None, parent_ids: list[str] | None = None,
               permissions: list[str] | None = None, application: str | None = None,
               tool: str | None = None, resource: str | None = None,
               outcome: str = "ok", duration_ms: int = 0, loss_proxy: float = 0.0,
               availability_value: float = 0.0, payload: dict | None = None,
               tags: list[str] | None = None, synthetic: bool = False,
               confidence: float = 1.0) -> dict:
    value = {
        "event_id": event_id, "incident_id": incident_id, "source_id": source_id,
        "agent_id": agent_id, "kind": kind, "event_time": timestamp(seconds),
        "observed_time": timestamp(observed_seconds if observed_seconds is not None else seconds + 1),
        "sequence": sequence, "parent_ids": parent_ids or [], "permissions": permissions or [],
        "outcome": outcome, "duration_ms": duration_ms, "loss_proxy": loss_proxy,
        "availability_value": availability_value, "payload": payload or {}, "tags": tags or [],
        "synthetic": synthetic, "confidence": confidence,
    }
    if correlation_id is not None:
        value["correlation_id"] = correlation_id
    if application is not None:
        value["application"] = application
    if tool is not None:
        value["tool"] = tool
    if resource is not None:
        value["resource"] = resource
    return value


def event(*args, **kwargs) -> Event:
    return Event.from_dict(event_dict(*args, **kwargs))


def incident_events() -> list[Event]:
    raw = [
        event_dict("e01", 1, "agent_started", 0),
        event_dict("e02", 2, "task_received", 1, correlation_id="task-1", parent_ids=["e01"]),
        event_dict("e03", 3, "permission_requested", 2, correlation_id="call-1", parent_ids=["e02"],
                   permissions=["calendar.write", "payments.send"]),
        event_dict("e04", 4, "permission_granted", 3, correlation_id="call-1", parent_ids=["e03"],
                   permissions=["calendar.write", "payments.send"]),
        event_dict("e05", 5, "tool_call_requested", 4, correlation_id="call-1", parent_ids=["e04"],
                   permissions=["payments.send"], application="wallet", tool="transfer"),
        event_dict("e06", 6, "tool_call_started", 5, correlation_id="call-1", parent_ids=["e05"],
                   permissions=["payments.send"], application="wallet", tool="transfer", loss_proxy=1.0,
                   availability_value=0.2),
        event_dict("e07", 7, "resource_written", 6, correlation_id="call-1", parent_ids=["e06"],
                   permissions=["payments.send"], application="wallet", tool="transfer",
                   resource="wallet:ledger:entry-77", loss_proxy=12.0, availability_value=1.0),
        event_dict("e08", 8, "tool_call_completed", 7, correlation_id="call-1", parent_ids=["e07"],
                   permissions=["payments.send"], application="wallet", tool="transfer", duration_ms=2100),
        event_dict("e09", 9, "tool_call_requested", 8, correlation_id="call-2", parent_ids=["e08"],
                   permissions=["messages.send"], application="chat", tool="send_message"),
        event_dict("e11", 11, "message_sent", 20, correlation_id="call-2", parent_ids=["e10"],
                   permissions=["messages.send"], application="chat", tool="send_message",
                   resource="chat:room:ops", loss_proxy=5.0, availability_value=0.4),
        event_dict("e12", 12, "tool_call_completed", 21, correlation_id="call-2", parent_ids=["e11"],
                   permissions=["messages.send"], application="chat", tool="send_message", duration_ms=13000),
        event_dict("e13", 13, "task_completed", 22, correlation_id="task-1", parent_ids=["e12"]),
        event_dict("e14", 14, "agent_stopped", 23, parent_ids=["e13"]),
    ]
    return [Event.from_dict(item) for item in raw]


def clean_events() -> list[Event]:
    values = incident_events()
    inserted = event("e10", 10, "tool_call_started", 9, correlation_id="call-2", parent_ids=["e09"],
                     permissions=["messages.send"], application="chat", tool="send_message")
    return sorted(values + [inserted], key=lambda item: item.sequence)


def policy(kind: str, policy_id: str | None = None, priority: int = 100, **parameters) -> PolicySpec:
    return PolicySpec(policy_id or kind, kind, priority, True, parameters)


def policy_scenarios() -> list[dict]:
    return [
        {"scenario_id": "baseline", "title": "无遏制基线", "policies": []},
        {"scenario_id": "permission", "title": "权限收缩", "policies": [
            {"policy_id": "p-permission", "kind": "permission_contraction", "priority": 10,
             "enabled": True, "parameters": {"forbidden_permissions": ["payments.send"], "mode": "block"}}]},
        {"scenario_id": "block", "title": "调用阻断", "policies": [
            {"policy_id": "p-block", "kind": "call_block", "priority": 10, "enabled": True,
             "parameters": {"tools": ["transfer"]}}]},
        {"scenario_id": "approval", "title": "审批门槛", "policies": [
            {"policy_id": "p-approval", "kind": "approval_gate", "priority": 10, "enabled": True,
             "parameters": {"permissions": ["payments.send", "messages.send"], "max_age_ms": 60000,
                            "review_delay_ms": 250}}]},
        {"scenario_id": "rate", "title": "速率限制", "policies": [
            {"policy_id": "p-rate", "kind": "rate_limit", "priority": 10, "enabled": True,
             "parameters": {"limit": 1, "window_ms": 10000, "scope": "agent", "mode": "delay"}}]},
        {"scenario_id": "isolation", "title": "应用隔离", "policies": [
            {"policy_id": "p-isolate", "kind": "application_isolation", "priority": 10, "enabled": True,
             "parameters": {"applications": ["wallet"], "allow_event_kinds": ["resource_read"]}}]},
        {"scenario_id": "combined", "title": "组合遏制", "policies": [
            {"policy_id": "p-combined-permission", "kind": "permission_contraction", "priority": 10,
             "enabled": True, "parameters": {"forbidden_permissions": ["payments.send"], "mode": "block"}},
            {"policy_id": "p-combined-rate", "kind": "rate_limit", "priority": 20, "enabled": True,
             "parameters": {"limit": 1, "window_ms": 5000, "scope": "agent", "mode": "delay"}},
            {"policy_id": "p-combined-isolate", "kind": "application_isolation", "priority": 30,
             "enabled": True, "parameters": {"applications": ["chat"], "allow_event_kinds": []}}]},
    ]


def fixture_dict() -> dict:
    events = [item.to_dict() for item in incident_events()]
    events = [events[0], events[2], events[1], events[5], events[4], events[3], events[6],
              events[8], events[7], events[9], events[10], events[12], events[11], events[4]]
    for item in events:
        item.pop("evidence_hash", None)
    return {
        "incident": {"incident_id": "inc-001", "title": "合成智能体越权调用事故",
                     "created_at": timestamp(0),
                     "metadata": {"environment": "offline-lab", "synthetic": True,
                                  "owner": "terminal-safety-team"}},
        "analysis": {"clock_skew_ms": 2000, "gap_threshold_ms": 8000,
                     "max_candidates_per_gap": 4, "confidence_floor": 0.15},
        "events": events,
        "checkpoints": [],
        "experiment": {"experiment_id": "exp-acceptance", "created_at": timestamp(40),
                       "seed": 20261112, "trials": 4, "uncertainty_mode": "sample_candidates",
                       "baseline_scenario": "baseline", "scenarios": policy_scenarios(),
                       "comparison_weights": {"loss_proxy": 0.55, "availability_cost": 0.2,
                                              "response_delay_ms": 0.1, "impact_agents": 0.05,
                                              "impact_applications": 0.05, "impact_resources": 0.05,
                                              "blocked_events": 0.0}},
    }


def checkpoint_from_state(checkpoint_id: str, state: ReplayState, seconds: int,
                          last_event_ids: tuple[str, ...]) -> Checkpoint:
    raw = state.to_dict()
    return Checkpoint(checkpoint_id, state.incident_id, timestamp(seconds), last_event_ids,
                      raw, digest_json(raw), "test")


class TempWorkspace:
    def __init__(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.path = Path(self._temp.name)

    def close(self) -> None:
        self._temp.cleanup()

    def write_fixture(self, fixture: dict | None = None) -> Path:
        path = self.path / "fixture.json"
        path.write_text(json.dumps(fixture or fixture_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
