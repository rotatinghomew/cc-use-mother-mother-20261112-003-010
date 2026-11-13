"""Composable offline containment policies and deterministic decisions."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import micros
from .errors import PolicyError
from .models import Decision, Event, PolicySpec, ReplayState


SUPPORTED_POLICIES = {
    "permission_contraction",
    "call_block",
    "approval_gate",
    "rate_limit",
    "application_isolation",
}


@dataclass
class PolicyRuntime:
    call_times: dict[str, deque[int]] = field(default_factory=lambda: defaultdict(deque))
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    isolated_apps: set[str] = field(default_factory=set)
    decisions: list[Decision] = field(default_factory=list)


class PolicySandbox:
    def __init__(self, specs: Iterable[PolicySpec]) -> None:
        self.specs = tuple(sorted((spec for spec in specs if spec.enabled), key=lambda item: (item.priority, item.policy_id)))
        unknown = [spec.kind for spec in self.specs if spec.kind not in SUPPORTED_POLICIES]
        if unknown:
            raise PolicyError("unknown containment policy", details={"kinds": unknown})

    def new_runtime(self) -> PolicyRuntime:
        runtime = PolicyRuntime()
        for spec in self.specs:
            if spec.kind == "application_isolation":
                runtime.isolated_apps.update(self._strings(spec.parameters, "applications"))
        return runtime

    def decide(self, event: Event, state: ReplayState, runtime: PolicyRuntime) -> Decision:
        matched: list[str] = []
        delay_ms = 0
        mutations: dict[str, Any] = {}
        allowed_actions: list[str] = []
        allowed_reasons: list[str] = []
        for spec in self.specs:
            if spec.kind == "permission_contraction":
                result = self._permission_contraction(spec, event, state)
            elif spec.kind == "call_block":
                result = self._call_block(spec, event)
            elif spec.kind == "approval_gate":
                result = self._approval_gate(spec, event, runtime)
            elif spec.kind == "rate_limit":
                result = self._rate_limit(spec, event, runtime)
            else:
                result = self._application_isolation(spec, event, runtime)
            if result is None:
                continue
            matched.append(spec.policy_id)
            delay_ms += result.get("delay_ms", 0)
            mutations.update(result.get("mutations", {}))
            if not result["allowed"]:
                decision = Decision(event.event_id, False, result["action"], tuple(matched), delay_ms,
                                    result["reason"], mutations)
                runtime.decisions.append(decision)
                return decision
            allowed_actions.append(result["action"])
            allowed_reasons.append(result["reason"])
        if not matched:
            action = "allow"
            reason = "no blocking policy matched"
        elif len(allowed_actions) == 1:
            action = allowed_actions[0]
            reason = allowed_reasons[0]
        else:
            action = "allow_with_controls"
            reason = "; ".join(allowed_reasons)
        decision = Decision(event.event_id, True, action,
                            tuple(matched), delay_ms, reason,
                            mutations)
        runtime.decisions.append(decision)
        return decision

    def observe(self, event: Event, decision: Decision, runtime: PolicyRuntime) -> None:
        if event.kind == "approval_granted" and event.correlation_id:
            runtime.approvals[event.correlation_id] = {
                "event_id": event.event_id,
                "event_time": event.event_time,
                "agent_id": event.agent_id,
                "application": event.application,
                "permissions": list(event.permissions),
            }
        if event.kind == "approval_denied" and event.correlation_id:
            runtime.approvals.pop(event.correlation_id, None)
        if event.kind in {"tool_call_requested", "tool_call_started"} and decision.allowed:
            event_time = micros(event.event_time)
            for scope in ("agent_tool", "agent", "application", "tool"):
                key = self._rate_key(event, scope)
                runtime.call_times[key].append(event_time)

    def _permission_contraction(self, spec: PolicySpec, event: Event, state: ReplayState) -> dict | None:
        forbidden = set(self._strings(spec.parameters, "forbidden_permissions"))
        allowed = set(self._strings(spec.parameters, "allowed_permissions"))
        agents = set(self._strings(spec.parameters, "agents"))
        if agents and event.agent_id not in agents:
            return None
        requested = set(event.permissions)
        if not requested:
            return None
        blocked = requested & forbidden
        if allowed:
            blocked |= requested - allowed
        if not blocked:
            return None
        mode = str(spec.parameters.get("mode", "block"))
        if mode == "shrink" and requested - blocked:
            return {"allowed": True, "action": "permission_shrink", "reason": "removed contracted permissions",
                    "mutations": {"effective_permissions": sorted(requested - blocked), "removed_permissions": sorted(blocked)}}
        return {"allowed": False, "action": "permission_block", "reason": "requested permission exceeds contracted set",
                "mutations": {"removed_permissions": sorted(blocked)}}

    def _call_block(self, spec: PolicySpec, event: Event) -> dict | None:
        kinds = set(self._strings(spec.parameters, "event_kinds"))
        tools = set(self._strings(spec.parameters, "tools"))
        applications = set(self._strings(spec.parameters, "applications"))
        tags = set(self._strings(spec.parameters, "tags"))
        permission_any = set(self._strings(spec.parameters, "permissions"))
        if kinds and event.kind not in kinds:
            return None
        matched = False
        matched |= bool(tools and event.tool in tools)
        matched |= bool(applications and event.application in applications)
        matched |= bool(tags and tags.intersection(event.tags))
        matched |= bool(permission_any and permission_any.intersection(event.permissions))
        if not any((tools, applications, tags, permission_any)):
            matched = event.kind in {"tool_call_requested", "tool_call_started"}
        if not matched:
            return None
        return {"allowed": False, "action": "call_block", "reason": "call matched explicit block rule"}

    def _approval_gate(self, spec: PolicySpec, event: Event, runtime: PolicyRuntime) -> dict | None:
        protected = set(self._strings(spec.parameters, "permissions"))
        tools = set(self._strings(spec.parameters, "tools"))
        if event.kind not in {"tool_call_requested", "tool_call_started", "resource_written", "resource_deleted", "message_sent"}:
            return None
        needs = bool(protected.intersection(event.permissions)) or bool(tools and event.tool in tools)
        if not needs:
            return None
        if not event.correlation_id:
            return {"allowed": False, "action": "approval_block", "reason": "protected action lacks correlation identifier"}
        approval = runtime.approvals.get(event.correlation_id)
        if not approval:
            return {"allowed": False, "action": "approval_block", "reason": "protected action lacks prior approval"}
        max_age_ms = int(spec.parameters.get("max_age_ms", 300000))
        age_ms = (micros(event.event_time) - micros(approval["event_time"])) // 1000
        if age_ms < 0 or age_ms > max_age_ms:
            return {"allowed": False, "action": "approval_block", "reason": "approval outside valid time window"}
        delay_ms = int(spec.parameters.get("review_delay_ms", 0))
        return {"allowed": True, "action": "approval_allow", "reason": "valid approval found", "delay_ms": delay_ms}

    def _rate_limit(self, spec: PolicySpec, event: Event, runtime: PolicyRuntime) -> dict | None:
        if event.kind not in {"tool_call_requested", "tool_call_started", "message_sent"}:
            return None
        limit = int(spec.parameters.get("limit", 1))
        window_ms = int(spec.parameters.get("window_ms", 1000))
        if limit <= 0 or window_ms <= 0:
            raise PolicyError("rate limit values must be positive", details={"policy_id": spec.policy_id})
        key = self._rate_key(event, str(spec.parameters.get("scope", "agent_tool")))
        now = micros(event.event_time)
        queue = runtime.call_times[key]
        cutoff = now - window_ms * 1000
        while queue and queue[0] <= cutoff:
            queue.popleft()
        if len(queue) < limit:
            return None
        mode = str(spec.parameters.get("mode", "block"))
        if mode == "delay":
            wait_ms = max(1, (queue[0] + window_ms * 1000 - now + 999) // 1000)
            return {"allowed": True, "action": "rate_delay", "reason": "rate budget exhausted; delayed",
                    "delay_ms": int(wait_ms)}
        return {"allowed": False, "action": "rate_block", "reason": "rate budget exhausted"}

    def _application_isolation(self, spec: PolicySpec, event: Event, runtime: PolicyRuntime) -> dict | None:
        if not event.application or event.application not in runtime.isolated_apps:
            return None
        allow_kinds = set(self._strings(spec.parameters, "allow_event_kinds"))
        allow_tools = set(self._strings(spec.parameters, "allow_tools"))
        if event.kind in allow_kinds or (event.tool and event.tool in allow_tools):
            return {"allowed": True, "action": "isolation_allow", "reason": "event allowed inside isolation"}
        if event.kind in {"application_isolated", "isolation_entered", "isolation_exited"}:
            return None
        return {"allowed": False, "action": "isolation_block", "reason": "application isolated from agent actions"}

    def _strings(self, parameters: dict, key: str) -> tuple[str, ...]:
        value = parameters.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise PolicyError("policy parameter must be string array", details={"field": key})
        return tuple(value)

    def _rate_key(self, event: Event, scope: str = "agent_tool") -> str:
        if scope == "application":
            return f"application:{event.application or '-'}"
        if scope == "agent":
            return f"agent:{event.agent_id}"
        if scope == "tool":
            return f"tool:{event.tool or '-'}"
        return f"agent_tool:{event.agent_id}:{event.tool or event.kind}"


def parse_policy_sets(raw: list[dict]) -> dict[str, tuple[PolicySpec, ...]]:
    result: dict[str, tuple[PolicySpec, ...]] = {}
    for entry in raw:
        scenario_id = str(entry.get("scenario_id", "")).strip()
        if not scenario_id:
            raise PolicyError("scenario_id is required")
        if scenario_id in result:
            raise PolicyError("duplicate scenario_id", details={"scenario_id": scenario_id})
        policies = entry.get("policies", [])
        if not isinstance(policies, list):
            raise PolicyError("policies must be an array", details={"scenario_id": scenario_id})
        result[scenario_id] = tuple(PolicySpec.from_dict(item) for item in policies)
    return result
