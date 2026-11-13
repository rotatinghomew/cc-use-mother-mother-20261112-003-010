"""Immutable domain records for evidence, replay, and experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .canonical import (
    digest_json,
    format_time,
    normalize_tags,
    optional_string,
    parse_time,
    require_bool,
    require_int,
    require_list,
    require_mapping,
    require_number,
    require_string,
)
from .errors import ContractError


EVENT_KINDS = {
    "agent_started", "agent_stopped", "task_received", "task_completed", "task_failed",
    "tool_call_requested", "tool_call_started", "tool_call_completed", "tool_call_failed",
    "permission_requested", "permission_granted", "permission_denied", "approval_requested",
    "approval_granted", "approval_denied", "resource_read", "resource_written",
    "resource_deleted", "message_sent", "message_received", "rate_limit_hit",
    "policy_applied", "isolation_entered", "isolation_exited", "checkpoint_created",
}
TERMINAL_KINDS = {"task_completed", "task_failed", "tool_call_completed", "tool_call_failed"}
MUTATION_KINDS = {"resource_written", "resource_deleted", "message_sent", "tool_call_started"}


@dataclass(frozen=True)
class Event:
    event_id: str
    incident_id: str
    source_id: str
    agent_id: str
    kind: str
    event_time: str
    observed_time: str
    sequence: int
    correlation_id: str | None = None
    parent_ids: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    application: str | None = None
    tool: str | None = None
    resource: str | None = None
    outcome: str = "unknown"
    duration_ms: int = 0
    loss_proxy: float = 0.0
    availability_value: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    synthetic: bool = False
    confidence: float = 1.0
    evidence_hash: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Event":
        if not isinstance(raw, dict):
            raise ContractError("event must be an object")
        kind = require_string(raw, "kind")
        if kind not in EVENT_KINDS:
            raise ContractError("unknown event kind", details={"kind": kind})
        event_time = require_string(raw, "event_time")
        observed_time = require_string(raw, "observed_time")
        parse_time(event_time)
        parse_time(observed_time)
        parent_ids = tuple(require_list(raw, "parent_ids", default=[]))
        permissions = tuple(require_list(raw, "permissions", default=[]))
        for field_name, values in (("parent_ids", parent_ids), ("permissions", permissions)):
            if any(not isinstance(value, str) or not value for value in values):
                raise ContractError("array must contain non-empty strings", details={"field": field_name})
        confidence = require_number(raw, "confidence", minimum=0.0, default=1.0)
        if confidence > 1.0:
            raise ContractError("confidence cannot exceed one")
        event = cls(
            event_id=require_string(raw, "event_id"),
            incident_id=require_string(raw, "incident_id"),
            source_id=require_string(raw, "source_id"),
            agent_id=require_string(raw, "agent_id"),
            kind=kind,
            event_time=event_time,
            observed_time=observed_time,
            sequence=require_int(raw, "sequence", minimum=0),
            correlation_id=optional_string(raw, "correlation_id"),
            parent_ids=parent_ids,
            permissions=permissions,
            application=optional_string(raw, "application"),
            tool=optional_string(raw, "tool"),
            resource=optional_string(raw, "resource"),
            outcome=require_string(raw, "outcome") if "outcome" in raw else "unknown",
            duration_ms=require_int(raw, "duration_ms", minimum=0, default=0),
            loss_proxy=require_number(raw, "loss_proxy", minimum=0.0, default=0.0),
            availability_value=require_number(raw, "availability_value", minimum=0.0, default=0.0),
            payload=require_mapping(raw, "payload", default={}),
            tags=normalize_tags(raw.get("tags")),
            synthetic=require_bool(raw, "synthetic", default=False),
            confidence=confidence,
            evidence_hash=optional_string(raw, "evidence_hash"),
        )
        expected = event.calculate_evidence_hash()
        if event.evidence_hash and event.evidence_hash != expected:
            raise ContractError("event evidence hash mismatch", details={"event_id": event.event_id})
        return replace(event, evidence_hash=expected)

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "incident_id": self.incident_id, "source_id": self.source_id,
            "agent_id": self.agent_id, "kind": self.kind, "event_time": format_time(parse_time(self.event_time)),
            "observed_time": format_time(parse_time(self.observed_time)), "sequence": self.sequence,
            "correlation_id": self.correlation_id, "parent_ids": list(self.parent_ids),
            "permissions": list(self.permissions), "application": self.application, "tool": self.tool,
            "resource": self.resource, "outcome": self.outcome, "duration_ms": self.duration_ms,
            "loss_proxy": self.loss_proxy, "availability_value": self.availability_value,
            "payload": self.payload, "tags": list(self.tags), "synthetic": self.synthetic,
            "confidence": self.confidence,
        }

    def calculate_evidence_hash(self) -> str:
        return digest_json(self.evidence_dict())

    def to_dict(self) -> dict[str, Any]:
        value = self.evidence_dict()
        value["evidence_hash"] = self.evidence_hash or self.calculate_evidence_hash()
        return value

    @property
    def is_mutation(self) -> bool:
        return self.kind in MUTATION_KINDS

    @property
    def is_terminal(self) -> bool:
        return self.kind in TERMINAL_KINDS


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    incident_id: str
    event_time: str
    last_event_ids: tuple[str, ...]
    state: dict[str, Any]
    state_hash: str
    source: str = "collector"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Checkpoint":
        if not isinstance(raw, dict):
            raise ContractError("checkpoint must be an object")
        event_time = require_string(raw, "event_time")
        parse_time(event_time)
        state = require_mapping(raw, "state")
        state_hash = require_string(raw, "state_hash") if "state_hash" in raw else digest_json(state)
        if state_hash != digest_json(state):
            raise ContractError("checkpoint state hash mismatch", details={"checkpoint_id": raw.get("checkpoint_id")})
        last_event_ids = tuple(require_list(raw, "last_event_ids", default=[]))
        if any(not isinstance(item, str) or not item for item in last_event_ids):
            raise ContractError("checkpoint last_event_ids must contain strings")
        return cls(require_string(raw, "checkpoint_id"), require_string(raw, "incident_id"), event_time,
                   last_event_ids, state, state_hash, require_string(raw, "source") if "source" in raw else "collector")

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint_id": self.checkpoint_id, "incident_id": self.incident_id,
                "event_time": format_time(parse_time(self.event_time)), "last_event_ids": list(self.last_event_ids),
                "state": self.state, "state_hash": self.state_hash, "source": self.source}


@dataclass(frozen=True)
class Gap:
    gap_id: str
    incident_id: str
    agent_id: str
    start_time: str
    end_time: str
    before_event_id: str | None
    after_event_id: str | None
    missing_sequences: tuple[int, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"gap_id": self.gap_id, "incident_id": self.incident_id, "agent_id": self.agent_id,
                "start_time": self.start_time, "end_time": self.end_time,
                "before_event_id": self.before_event_id, "after_event_id": self.after_event_id,
                "missing_sequences": list(self.missing_sequences), "reasons": list(self.reasons)}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    gap_id: str
    events: tuple[Event, ...]
    confidence: float
    rationale: tuple[str, ...]
    contradictions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "gap_id": self.gap_id,
                "events": [event.to_dict() for event in self.events], "confidence": self.confidence,
                "rationale": list(self.rationale), "contradictions": list(self.contradictions)}


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    kind: str
    priority: int
    enabled: bool
    parameters: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PolicySpec":
        if not isinstance(raw, dict):
            raise ContractError("policy must be an object")
        return cls(require_string(raw, "policy_id"), require_string(raw, "kind"),
                   require_int(raw, "priority", minimum=0, default=100),
                   require_bool(raw, "enabled", default=True), require_mapping(raw, "parameters", default={}))

    def to_dict(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "kind": self.kind, "priority": self.priority,
                "enabled": self.enabled, "parameters": self.parameters}


@dataclass(frozen=True)
class Decision:
    event_id: str
    allowed: bool
    action: str
    policy_ids: tuple[str, ...]
    delay_ms: int
    reason: str
    mutations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "allowed": self.allowed, "action": self.action,
                "policy_ids": list(self.policy_ids), "delay_ms": self.delay_ms,
                "reason": self.reason, "mutations": self.mutations}


@dataclass
class ReplayState:
    incident_id: str
    active_agents: set[str] = field(default_factory=set)
    permissions: dict[str, set[str]] = field(default_factory=dict)
    pending_approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    applications: dict[str, str] = field(default_factory=dict)
    resources: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    tasks: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    total_loss: float = 0.0
    total_availability: float = 0.0
    total_delay_ms: int = 0
    contained_events: int = 0
    blocked_events: int = 0
    applied_events: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReplayState":
        state = cls(incident_id=str(raw.get("incident_id", "unknown")))
        state.active_agents = set(raw.get("active_agents", []))
        state.permissions = {key: set(value) for key, value in raw.get("permissions", {}).items()}
        state.pending_approvals = dict(raw.get("pending_approvals", {}))
        state.applications = dict(raw.get("applications", {}))
        state.resources = dict(raw.get("resources", {}))
        state.open_calls = dict(raw.get("open_calls", {}))
        state.tasks = dict(raw.get("tasks", {}))
        state.counters = {key: int(value) for key, value in raw.get("counters", {}).items()}
        state.timeline = list(raw.get("timeline", []))
        state.total_loss = float(raw.get("total_loss", 0.0))
        state.total_availability = float(raw.get("total_availability", 0.0))
        state.total_delay_ms = int(raw.get("total_delay_ms", 0))
        state.contained_events = int(raw.get("contained_events", 0))
        state.blocked_events = int(raw.get("blocked_events", 0))
        state.applied_events = int(raw.get("applied_events", 0))
        return state

    def to_dict(self) -> dict[str, Any]:
        return {"incident_id": self.incident_id, "active_agents": sorted(self.active_agents),
                "permissions": {key: sorted(value) for key, value in sorted(self.permissions.items())},
                "pending_approvals": self.pending_approvals, "applications": self.applications,
                "resources": self.resources, "open_calls": self.open_calls, "tasks": self.tasks,
                "counters": self.counters, "timeline": self.timeline, "total_loss": self.total_loss,
                "total_availability": self.total_availability, "total_delay_ms": self.total_delay_ms,
                "contained_events": self.contained_events, "blocked_events": self.blocked_events,
                "applied_events": self.applied_events}

    def state_hash(self) -> str:
        return digest_json(self.to_dict())


@dataclass(frozen=True)
class RunMetrics:
    impact_agents: int
    impact_applications: int
    impact_resources: int
    loss_proxy: float
    availability_cost: float
    response_delay_ms: int
    blocked_events: int
    contained_events: int
    applied_events: int

    def to_dict(self) -> dict[str, Any]:
        return {"impact_agents": self.impact_agents, "impact_applications": self.impact_applications,
                "impact_resources": self.impact_resources, "loss_proxy": self.loss_proxy,
                "availability_cost": self.availability_cost, "response_delay_ms": self.response_delay_ms,
                "blocked_events": self.blocked_events, "contained_events": self.contained_events,
                "applied_events": self.applied_events}


def event_index(events: Iterable[Event]) -> dict[str, Event]:
    result: dict[str, Event] = {}
    for event in events:
        if event.event_id in result and result[event.event_id].evidence_hash != event.evidence_hash:
            raise ContractError("conflicting event identifiers", details={"event_id": event.event_id})
        result[event.event_id] = event
    return result
