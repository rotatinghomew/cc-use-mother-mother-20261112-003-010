"""Cross-artifact invariant checks for replay and experiment results."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .canonical import digest_json, micros
from .experiments import ExperimentResult
from .models import Event
from .replay import ReplayResult


@dataclass(frozen=True)
class InvariantViolation:
    code: str
    message: str
    context: dict

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "context": self.context}


class InvariantChecker:
    def check_replay(self, result: ReplayResult, evidence: Iterable[Event]) -> dict:
        events = list(evidence)
        violations: list[InvariantViolation] = []
        violations.extend(self._event_membership(result, events))
        violations.extend(self._decision_alignment(result))
        violations.extend(self._timeline_order(result))
        violations.extend(self._metric_alignment(result))
        violations.extend(self._state_closure(result))
        material = [item.to_dict() for item in violations]
        return {"ok": not violations, "run_id": result.run_id, "violations": material,
                "check_hash": digest_json({"run_id": result.run_id, "violations": material})}

    def check_experiment(self, result: ExperimentResult, evidence: Iterable[Event]) -> dict:
        events = list(evidence)
        trial_checks = [self.check_replay(trial.replay, events) for trial in result.trials]
        violations: list[InvariantViolation] = []
        expected_trials = result.plan.trials * len(result.plan.scenarios)
        if len(result.trials) != expected_trials:
            violations.append(InvariantViolation(
                "trial_count_mismatch", "实验试验数与计划不一致",
                {"expected": expected_trials, "actual": len(result.trials)},
            ))
        scenario_counts = Counter(trial.scenario_id for trial in result.trials)
        for scenario in result.plan.scenarios:
            actual = scenario_counts.get(scenario.scenario_id, 0)
            if actual != result.plan.trials:
                violations.append(InvariantViolation(
                    "scenario_trial_count", "场景试验数与计划不一致",
                    {"scenario_id": scenario.scenario_id, "expected": result.plan.trials, "actual": actual},
                ))
        trial_ids = [trial.trial_id for trial in result.trials]
        if len(trial_ids) != len(set(trial_ids)):
            violations.append(InvariantViolation(
                "duplicate_trial_id", "实验内存在重复试验标识", {"trial_ids": trial_ids},
            ))
        summary_ids = set(result.scenario_summaries)
        planned_ids = {scenario.scenario_id for scenario in result.plan.scenarios}
        if summary_ids != planned_ids:
            violations.append(InvariantViolation(
                "summary_scenario_mismatch", "汇总场景与计划场景不一致",
                {"planned": sorted(planned_ids), "summary": sorted(summary_ids)},
            ))
        failed_trials = [check for check in trial_checks if not check["ok"]]
        if failed_trials:
            violations.append(InvariantViolation(
                "trial_invariant_failure", "至少一个试验违反重放不变量",
                {"run_ids": [check["run_id"] for check in failed_trials]},
            ))
        material = [item.to_dict() for item in violations]
        return {"ok": not violations, "experiment_id": result.plan.experiment_id,
                "violations": material, "trial_checks": trial_checks,
                "check_hash": digest_json({"experiment_id": result.plan.experiment_id,
                                           "violations": material,
                                           "trial_hashes": [check["check_hash"] for check in trial_checks]})}

    def _event_membership(self, result: ReplayResult, events: list[Event]) -> list[InvariantViolation]:
        known = {event.event_id for event in events}
        processed = set(result.processed_event_ids)
        skipped = set(result.skipped_event_ids)
        violations: list[InvariantViolation] = []
        overlap = sorted(processed & skipped)
        if overlap:
            violations.append(InvariantViolation(
                "processed_skipped_overlap", "事件同时出现在已处理与跳过集合",
                {"event_ids": overlap},
            ))
        unknown = sorted((processed | skipped) - known)
        if unknown:
            violations.append(InvariantViolation(
                "unknown_result_event", "重放结果引用了输入流中不存在的事件",
                {"event_ids": unknown},
            ))
        uncovered = sorted(known - processed - skipped)
        if uncovered:
            violations.append(InvariantViolation(
                "unaccounted_event", "输入事件未被处理或显式跳过",
                {"event_ids": uncovered},
            ))
        return violations

    def _decision_alignment(self, result: ReplayResult) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        decision_ids = [decision.event_id for decision in result.decisions]
        timeline_ids = [item["event_id"] for item in result.state.timeline]
        if decision_ids != timeline_ids:
            violations.append(InvariantViolation(
                "decision_timeline_mismatch", "策略决策顺序与时间线顺序不一致",
                {"decision_ids": decision_ids, "timeline_ids": timeline_ids},
            ))
        blocked = sorted(decision.event_id for decision in result.decisions if not decision.allowed)
        missing_from_skipped = sorted(set(blocked) - set(result.skipped_event_ids))
        if missing_from_skipped:
            violations.append(InvariantViolation(
                "blocked_not_skipped", "被阻断事件未记录为跳过",
                {"event_ids": missing_from_skipped},
            ))
        allowed = sorted(decision.event_id for decision in result.decisions if decision.allowed)
        missing_from_processed = sorted(set(allowed) - set(result.processed_event_ids))
        if missing_from_processed:
            violations.append(InvariantViolation(
                "allowed_not_processed", "允许事件未记录为已处理",
                {"event_ids": missing_from_processed},
            ))
        return violations

    def _timeline_order(self, result: ReplayResult) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        timeline = result.state.timeline
        event_times = [micros(item["event_time"]) for item in timeline]
        if event_times != sorted(event_times):
            violations.append(InvariantViolation(
                "timeline_not_event_time_ordered", "时间线未按事件时间单调排列",
                {"event_ids": [item["event_id"] for item in timeline]},
            ))
        losses = [float(item["cumulative_loss"]) for item in timeline]
        if any(losses[index] > losses[index + 1] for index in range(len(losses) - 1)):
            violations.append(InvariantViolation(
                "loss_not_monotonic", "累计损失代理出现下降",
                {"values": losses},
            ))
        delays = [int(item["cumulative_delay_ms"]) for item in timeline]
        if any(delays[index] > delays[index + 1] for index in range(len(delays) - 1)):
            violations.append(InvariantViolation(
                "delay_not_monotonic", "累计响应时延出现下降",
                {"values": delays},
            ))
        return violations

    def _metric_alignment(self, result: ReplayResult) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        metrics = result.metrics
        state = result.state
        comparisons = {
            "loss_proxy": (metrics.loss_proxy, round(state.total_loss, 6)),
            "availability_cost": (metrics.availability_cost, round(state.total_availability, 6)),
            "response_delay_ms": (metrics.response_delay_ms, state.total_delay_ms),
            "blocked_events": (metrics.blocked_events, state.blocked_events),
            "contained_events": (metrics.contained_events, state.contained_events),
            "applied_events": (metrics.applied_events, state.applied_events),
        }
        for field, (reported, actual) in comparisons.items():
            if reported != actual:
                violations.append(InvariantViolation(
                    "metric_state_mismatch", "汇总指标与重放状态不一致",
                    {"field": field, "reported": reported, "actual": actual},
                ))
        if result.state_hash != state.state_hash():
            violations.append(InvariantViolation(
                "state_hash_mismatch", "结果状态哈希与当前状态不一致",
                {"reported": result.state_hash, "actual": state.state_hash()},
            ))
        return violations

    def _state_closure(self, result: ReplayResult) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        terminal_tasks = {key for key, value in result.state.tasks.items() if value in {"completed", "failed"}}
        open_task_calls = sorted(key for key in result.state.open_calls if key in terminal_tasks)
        if open_task_calls:
            violations.append(InvariantViolation(
                "terminal_task_has_open_call", "已终结任务仍保留打开的调用",
                {"correlation_ids": open_task_calls},
            ))
        negative_counters = {key: value for key, value in result.state.counters.items() if value < 0}
        if negative_counters:
            violations.append(InvariantViolation(
                "negative_counter", "状态计数器不得为负数", {"counters": negative_counters},
            ))
        invalid_resources = sorted(key for key, value in result.state.resources.items()
                                   if value.get("writes", 0) < 0 or value.get("deletes", 0) < 0
                                   or value.get("reads", 0) < 0)
        if invalid_resources:
            violations.append(InvariantViolation(
                "invalid_resource_counter", "资源读写计数不得为负数",
                {"resources": invalid_resources},
            ))
        return violations
