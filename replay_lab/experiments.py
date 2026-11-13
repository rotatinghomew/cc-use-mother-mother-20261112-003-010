"""Deterministic experiment planning across policies and uncertain gaps."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .canonical import deterministic_id, digest_json, format_time, mean, parse_time, percentile
from .errors import ExperimentError
from .interpolation import Interpolator
from .models import Candidate, Checkpoint, Event, PolicySpec
from .policies import PolicySandbox
from .replay import ReplayEngine, ReplayResult


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    policies: tuple[PolicySpec, ...]

    def to_dict(self) -> dict:
        return {"scenario_id": self.scenario_id, "title": self.title,
                "policies": [policy.to_dict() for policy in self.policies]}


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    incident_id: str
    created_at: str
    seed: int
    trials: int
    uncertainty_mode: str
    scenarios: tuple[Scenario, ...]
    parameters: dict

    def to_dict(self) -> dict:
        return {"experiment_id": self.experiment_id, "incident_id": self.incident_id,
                "created_at": self.created_at, "seed": self.seed, "trials": self.trials,
                "uncertainty_mode": self.uncertainty_mode,
                "scenarios": [scenario.to_dict() for scenario in self.scenarios],
                "parameters": self.parameters}


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    trial_index: int
    trial_seed: int
    scenario_id: str
    selected_candidate_ids: tuple[str, ...]
    replay: ReplayResult
    trial_hash: str

    def to_dict(self) -> dict:
        return {"trial_id": self.trial_id, "trial_index": self.trial_index,
                "trial_seed": self.trial_seed, "scenario_id": self.scenario_id,
                "selected_candidate_ids": list(self.selected_candidate_ids),
                "replay": self.replay.to_dict(), "trial_hash": self.trial_hash}


@dataclass(frozen=True)
class ExperimentResult:
    plan: ExperimentPlan
    trials: tuple[TrialResult, ...]
    scenario_summaries: dict[str, dict]
    experiment_hash: str

    def to_dict(self) -> dict:
        return {"plan": self.plan.to_dict(), "trials": [trial.to_dict() for trial in self.trials],
                "scenario_summaries": self.scenario_summaries, "experiment_hash": self.experiment_hash}


class ExperimentRunner:
    MODES = {"evidence_only", "best_candidate", "sample_candidates", "all_candidates"}

    def __init__(self, engine: ReplayEngine | None = None, interpolator: Interpolator | None = None) -> None:
        self.engine = engine or ReplayEngine()
        self.interpolator = interpolator or Interpolator()

    def build_plan(self, *, incident_id: str, created_at: str, seed: int, trials: int,
                   uncertainty_mode: str, scenarios: Iterable[Scenario], parameters: dict | None = None,
                   experiment_id: str | None = None) -> ExperimentPlan:
        parse_time(created_at)
        if uncertainty_mode not in self.MODES:
            raise ExperimentError("unknown uncertainty mode", details={"mode": uncertainty_mode})
        if trials < 1 or trials > 1000:
            raise ExperimentError("trials must be between one and one thousand")
        scenario_values = tuple(scenarios)
        if not scenario_values:
            raise ExperimentError("at least one scenario is required")
        ids = [scenario.scenario_id for scenario in scenario_values]
        if len(ids) != len(set(ids)):
            raise ExperimentError("scenario identifiers must be unique")
        identity = {"incident_id": incident_id, "created_at": created_at, "seed": seed,
                    "trials": trials, "uncertainty_mode": uncertainty_mode,
                    "scenarios": [scenario.to_dict() for scenario in scenario_values],
                    "parameters": parameters or {}}
        return ExperimentPlan(experiment_id or deterministic_id("experiment", identity), incident_id,
                              created_at, seed, trials, uncertainty_mode, scenario_values, parameters or {})

    def run(self, plan: ExperimentPlan, evidence: Iterable[Event], candidates: Iterable[Candidate],
            *, checkpoint: Checkpoint | None = None) -> ExperimentResult:
        events = list(evidence)
        if any(event.incident_id != plan.incident_id for event in events):
            raise ExperimentError("evidence incident does not match plan")
        candidate_values = list(candidates)
        by_gap: dict[str, list[Candidate]] = {}
        for candidate in candidate_values:
            by_gap.setdefault(candidate.gap_id, []).append(candidate)
        for values in by_gap.values():
            values.sort(key=lambda item: (-item.confidence, item.candidate_id))
        trials: list[TrialResult] = []
        for trial_index in range(plan.trials):
            trial_seed = self._trial_seed(plan.seed, trial_index)
            selected = self._select_candidates(plan.uncertainty_mode, by_gap, trial_seed, trial_index)
            overlay = list(events)
            for candidate in selected:
                overlay.extend(candidate.events)
            for scenario in plan.scenarios:
                sandbox = PolicySandbox(scenario.policies)
                replay = self.engine.replay(overlay, scenario_id=scenario.scenario_id, policies=sandbox,
                                            checkpoint=checkpoint, include_synthetic=True)
                identity = {"experiment_id": plan.experiment_id, "trial_index": trial_index,
                            "trial_seed": trial_seed, "scenario_id": scenario.scenario_id,
                            "candidate_ids": [candidate.candidate_id for candidate in selected],
                            "replay_hash": replay.state_hash, "stream_hash": replay.event_stream_hash}
                trial_hash = digest_json(identity)
                trial_id = deterministic_id("trial", identity)
                trials.append(TrialResult(trial_id, trial_index, trial_seed, scenario.scenario_id,
                                          tuple(candidate.candidate_id for candidate in selected), replay, trial_hash))
        summaries = self._summarize(plan, trials)
        result_identity = {"plan": plan.to_dict(), "trial_hashes": [trial.trial_hash for trial in trials],
                           "summaries": summaries}
        return ExperimentResult(plan, tuple(trials), summaries, digest_json(result_identity))

    def _trial_seed(self, seed: int, index: int) -> int:
        return int(digest_json({"seed": seed, "trial": index})[:16], 16)

    def _select_candidates(self, mode: str, by_gap: dict[str, list[Candidate]], seed: int,
                           trial_index: int) -> list[Candidate]:
        if mode == "evidence_only":
            return []
        if mode == "best_candidate":
            return [values[0] for _, values in sorted(by_gap.items()) if values]
        if mode == "all_candidates":
            return [candidate for _, values in sorted(by_gap.items()) for candidate in values]
        rng = random.Random(seed)
        selected: list[Candidate] = []
        for gap_id, values in sorted(by_gap.items()):
            if not values:
                continue
            weights = [max(candidate.confidence, 0.001) for candidate in values]
            none_weight = max(0.05, 1.0 - max(weights))
            population: list[Candidate | None] = values + [None]
            chosen = rng.choices(population, weights=weights + [none_weight], k=1)[0]
            if chosen:
                selected.append(chosen)
        return selected

    def _summarize(self, plan: ExperimentPlan, trials: list[TrialResult]) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for scenario in plan.scenarios:
            values = [trial for trial in trials if trial.scenario_id == scenario.scenario_id]
            metrics = [trial.replay.metrics for trial in values]
            result[scenario.scenario_id] = {
                "title": scenario.title,
                "trials": len(values),
                "state_hashes": sorted({trial.replay.state_hash for trial in values}),
                "candidate_sets": sorted({tuple(trial.selected_candidate_ids) for trial in values}),
                "loss_proxy": self._distribution([metric.loss_proxy for metric in metrics]),
                "availability_cost": self._distribution([metric.availability_cost for metric in metrics]),
                "response_delay_ms": self._distribution([float(metric.response_delay_ms) for metric in metrics]),
                "impact_agents": self._distribution([float(metric.impact_agents) for metric in metrics]),
                "impact_applications": self._distribution([float(metric.impact_applications) for metric in metrics]),
                "impact_resources": self._distribution([float(metric.impact_resources) for metric in metrics]),
                "blocked_events": self._distribution([float(metric.blocked_events) for metric in metrics]),
                "contained_events": self._distribution([float(metric.contained_events) for metric in metrics]),
            }
        return result

    def _distribution(self, values: list[float]) -> dict:
        return {"min": min(values) if values else 0.0, "mean": mean(values),
                "p50": percentile(values, 0.5), "p90": percentile(values, 0.9),
                "max": max(values) if values else 0.0}


def scenarios_from_dict(raw: list[dict]) -> tuple[Scenario, ...]:
    result: list[Scenario] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ExperimentError("scenario must be an object")
        scenario_id = str(item.get("scenario_id", "")).strip()
        title = str(item.get("title", scenario_id)).strip()
        policies = item.get("policies", [])
        if not scenario_id or not isinstance(policies, list):
            raise ExperimentError("scenario requires identifier and policy array")
        result.append(Scenario(scenario_id, title, tuple(PolicySpec.from_dict(value) for value in policies)))
    return tuple(result)
