"""Explainable fixture, policy, and replay diagnostics for operators."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import digest_json, micros
from .models import Candidate, Event, Gap, PolicySpec


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    subjects: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "message": self.message,
                "subjects": list(self.subjects), "evidence": self.evidence}


class Diagnostics:
    def coverage_matrix(self, events: Iterable[Event]) -> dict[str, Any]:
        """Summarize evidentiary coverage and expose replay blind spots."""
        values = list(events)
        sources: dict[str, Counter] = defaultdict(Counter)
        agents: dict[str, Counter] = defaultdict(Counter)
        applications: dict[str, Counter] = defaultdict(Counter)
        permission_sources: dict[str, set[str]] = defaultdict(set)
        mutations_without_resource: list[str] = []
        terminal_without_correlation: list[str] = []
        low_confidence_events: list[str] = []
        for event in values:
            sources[event.source_id][event.kind] += 1
            agents[event.agent_id][event.kind] += 1
            application = event.application or "unattributed"
            applications[application][event.kind] += 1
            for permission in event.permissions:
                permission_sources[permission].add(event.source_id)
            if event.is_mutation and not event.resource:
                mutations_without_resource.append(event.event_id)
            if event.is_terminal and not event.correlation_id:
                terminal_without_correlation.append(event.event_id)
            if event.confidence < 0.75:
                low_confidence_events.append(event.event_id)
        kinds = sorted({event.kind for event in values})
        source_rows = []
        for source_id, counts in sorted(sources.items()):
            missing = [kind for kind in kinds if counts[kind] == 0]
            source_rows.append({
                "source_id": source_id,
                "event_count": sum(counts.values()),
                "kinds": dict(sorted(counts.items())),
                "missing_kinds": missing,
                "coverage_ratio": round((len(kinds) - len(missing)) / len(kinds), 6) if kinds else 1.0,
            })
        single_source_permissions = {
            permission: sorted(source_ids)
            for permission, source_ids in sorted(permission_sources.items())
            if len(source_ids) == 1
        }
        blind_spots = {
            "mutations_without_resource": sorted(mutations_without_resource),
            "terminal_without_correlation": sorted(terminal_without_correlation),
            "low_confidence_events": sorted(low_confidence_events),
            "single_source_permissions": single_source_permissions,
        }
        severity = "critical" if mutations_without_resource else (
            "warning" if terminal_without_correlation or low_confidence_events or single_source_permissions else "clear"
        )
        return {
            "event_count": len(values),
            "event_kinds": kinds,
            "sources": source_rows,
            "agents": {key: dict(sorted(value.items())) for key, value in sorted(agents.items())},
            "applications": {key: dict(sorted(value.items())) for key, value in sorted(applications.items())},
            "blind_spots": blind_spots,
            "severity": severity,
            "coverage_hash": digest_json({"sources": source_rows, "blind_spots": blind_spots}),
        }

    def fixture_findings(self, events: Iterable[Event], gaps: Iterable[Gap],
                         candidates: Iterable[Candidate]) -> list[Finding]:
        values = list(events)
        gap_values = list(gaps)
        candidate_values = list(candidates)
        findings: list[Finding] = []
        findings.extend(self._clock_findings(values))
        findings.extend(self._sequence_findings(values))
        findings.extend(self._coverage_findings(values))
        findings.extend(self._gap_findings(gap_values, candidate_values))
        findings.extend(self._risk_findings(values))
        return sorted(findings, key=lambda item: (self._severity_rank(item.severity), item.code, item.subjects))

    def policy_findings(self, policies: Iterable[PolicySpec], events: Iterable[Event]) -> list[Finding]:
        specs = list(policies)
        values = list(events)
        findings: list[Finding] = []
        ids = Counter(spec.policy_id for spec in specs)
        for policy_id, count in sorted(ids.items()):
            if count > 1:
                findings.append(Finding("duplicate_policy_id", "error", "策略标识重复",
                                        (policy_id,), {"count": count}))
        priorities: dict[int, list[str]] = defaultdict(list)
        for spec in specs:
            priorities[spec.priority].append(spec.policy_id)
            findings.extend(self._single_policy_findings(spec, values))
        for priority, policy_ids in sorted(priorities.items()):
            if len(policy_ids) > 1:
                findings.append(Finding("shared_priority", "info", "同优先级按策略标识稳定排序",
                                        tuple(sorted(policy_ids)), {"priority": priority}))
        protected_permissions = {permission for event in values if event.loss_proxy > 0 for permission in event.permissions}
        gated_permissions = {permission for spec in specs if spec.kind == "approval_gate"
                             for permission in spec.parameters.get("permissions", [])}
        contracted_permissions = {permission for spec in specs if spec.kind == "permission_contraction"
                                  for permission in spec.parameters.get("forbidden_permissions", [])}
        uncovered = protected_permissions - gated_permissions - contracted_permissions
        if uncovered:
            findings.append(Finding("risky_permission_uncovered", "warning", "高损失事件权限未被审批或收缩策略覆盖",
                                    tuple(sorted(uncovered)), {"permissions": sorted(uncovered)}))
        return sorted(findings, key=lambda item: (self._severity_rank(item.severity), item.code, item.subjects))

    def experiment_acceptance(self, summary: dict, *, max_loss: float | None = None,
                              max_availability: float | None = None,
                              max_delay_ms: float | None = None) -> dict:
        checks: list[dict] = []
        for scenario_id, metrics in sorted(summary.items()):
            scenario_checks: list[dict] = []
            if max_loss is not None:
                scenario_checks.append(self._check("loss_proxy", metrics["loss_proxy"]["mean"], "<=", max_loss))
            if max_availability is not None:
                scenario_checks.append(self._check("availability_cost", metrics["availability_cost"]["mean"],
                                                   "<=", max_availability))
            if max_delay_ms is not None:
                scenario_checks.append(self._check("response_delay_ms", metrics["response_delay_ms"]["mean"],
                                                   "<=", max_delay_ms))
            checks.append({"scenario_id": scenario_id, "ok": all(check["ok"] for check in scenario_checks),
                           "checks": scenario_checks})
        return {"ok": all(item["ok"] for item in checks), "scenarios": checks,
                "criteria_hash": digest_json({"max_loss": max_loss, "max_availability": max_availability,
                                              "max_delay_ms": max_delay_ms})}

    def evidence_preservation(self, before: Iterable[Event], after: Iterable[Event]) -> dict:
        left = {event.event_id: event.evidence_hash for event in before if not event.synthetic}
        right = {event.event_id: event.evidence_hash for event in after if not event.synthetic}
        missing = sorted(set(left) - set(right))
        added_real = sorted(set(right) - set(left))
        changed = sorted(event_id for event_id in set(left) & set(right) if left[event_id] != right[event_id])
        return {"preserved": not missing and not added_real and not changed,
                "missing_evidence_ids": missing, "added_evidence_ids": added_real,
                "changed_evidence_ids": changed, "before_hash": digest_json(left), "after_hash": digest_json(right)}

    def _clock_findings(self, events: list[Event]) -> list[Finding]:
        result: list[Finding] = []
        for event in events:
            skew_ms = (micros(event.observed_time) - micros(event.event_time)) / 1000
            if skew_ms < 0:
                result.append(Finding("negative_observation_lag", "warning", "接收时间早于事件时间",
                                      (event.event_id,), {"lag_ms": skew_ms}))
            elif skew_ms > 60000:
                result.append(Finding("large_observation_lag", "info", "事件存在较大迟到接收",
                                      (event.event_id,), {"lag_ms": skew_ms}))
        return result

    def _sequence_findings(self, events: list[Event]) -> list[Finding]:
        groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
        for event in events:
            groups[(event.source_id, event.agent_id)].append(event)
        result: list[Finding] = []
        for (source_id, agent_id), values in sorted(groups.items()):
            sequences = sorted(event.sequence for event in values)
            duplicates = [number for number, count in Counter(sequences).items() if count > 1]
            if duplicates:
                result.append(Finding("duplicate_sequence", "error", "同一来源与智能体存在重复序号",
                                      (source_id, agent_id), {"sequences": duplicates}))
            if sequences:
                expected = set(range(sequences[0], sequences[-1] + 1))
                missing = sorted(expected - set(sequences))
                if missing:
                    result.append(Finding("missing_sequence", "warning", "来源序号存在缺口",
                                          (source_id, agent_id), {"sequences": missing}))
        return result

    def _coverage_findings(self, events: list[Event]) -> list[Finding]:
        result: list[Finding] = []
        agents = sorted({event.agent_id for event in events})
        for agent_id in agents:
            values = [event for event in events if event.agent_id == agent_id]
            kinds = {event.kind for event in values}
            if "agent_started" not in kinds:
                result.append(Finding("missing_agent_start", "info", "智能体缺少启动事件",
                                      (agent_id,), {"event_count": len(values)}))
            correlations = defaultdict(list)
            for event in values:
                if event.correlation_id:
                    correlations[event.correlation_id].append(event)
            for correlation_id, group in sorted(correlations.items()):
                if any(event.kind == "tool_call_requested" for event in group) and not any(event.is_terminal for event in group):
                    result.append(Finding("open_tool_call", "warning", "工具调用缺少终态",
                                          (agent_id, correlation_id),
                                          {"event_ids": [event.event_id for event in group]}))
        return result

    def _gap_findings(self, gaps: list[Gap], candidates: list[Candidate]) -> list[Finding]:
        by_gap: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            by_gap[candidate.gap_id].append(candidate)
        result: list[Finding] = []
        for gap in gaps:
            values = by_gap.get(gap.gap_id, [])
            if not values:
                result.append(Finding("gap_without_candidate", "warning", "缺口没有可接受的插值候选",
                                      (gap.gap_id,), {"reasons": list(gap.reasons)}))
            elif max(candidate.confidence for candidate in values) < 0.5:
                result.append(Finding("low_confidence_gap", "info", "缺口仅有低置信候选",
                                      (gap.gap_id,), {"max_confidence": max(candidate.confidence for candidate in values)}))
        return result

    def _risk_findings(self, events: list[Event]) -> list[Finding]:
        result: list[Finding] = []
        high = [event for event in events if event.loss_proxy >= 5.0]
        for event in high:
            if event.kind in {"resource_written", "resource_deleted", "message_sent", "tool_call_started"}:
                result.append(Finding("high_loss_mutation", "warning", "高损失代理事件会改变外部或持久状态",
                                      (event.event_id,), {"loss_proxy": event.loss_proxy,
                                                         "permissions": list(event.permissions)}))
        return result

    def _single_policy_findings(self, spec: PolicySpec, events: list[Event]) -> list[Finding]:
        result: list[Finding] = []
        if not spec.enabled:
            return [Finding("disabled_policy", "info", "策略在场景中被禁用", (spec.policy_id,), {})]
        if spec.kind == "rate_limit":
            if int(spec.parameters.get("limit", 1)) <= 0 or int(spec.parameters.get("window_ms", 1000)) <= 0:
                result.append(Finding("invalid_rate_limit", "error", "速率限制必须使用正数",
                                      (spec.policy_id,), spec.parameters))
        elif spec.kind == "application_isolation":
            applications = set(spec.parameters.get("applications", []))
            observed = {event.application for event in events if event.application}
            if applications and not applications.intersection(observed):
                result.append(Finding("isolation_matches_nothing", "info", "隔离策略未匹配夹具应用",
                                      (spec.policy_id,), {"applications": sorted(applications)}))
        elif spec.kind == "call_block":
            if not any(spec.parameters.get(key) for key in ("tools", "applications", "tags", "permissions", "event_kinds")):
                result.append(Finding("broad_call_block", "warning", "调用阻断未限定匹配面",
                                      (spec.policy_id,), {}))
        return result

    def _check(self, metric: str, actual: float, operator: str, threshold: float) -> dict:
        return {"metric": metric, "actual": actual, "operator": operator, "threshold": threshold,
                "ok": actual <= threshold}

    def _severity_rank(self, severity: str) -> int:
        return {"error": 0, "warning": 1, "info": 2}.get(severity, 3)
