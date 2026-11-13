"""Local sensitivity analysis for containment thresholds and uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import deterministic_id, digest_json
from .experiments import ExperimentRunner, Scenario
from .models import Candidate, Event, PolicySpec


@dataclass(frozen=True)
class SensitivityPoint:
    point_id: str
    scenario_id: str
    parameter_path: str
    parameter_value: Any
    metrics: dict
    experiment_hash: str

    def to_dict(self) -> dict:
        return {"point_id": self.point_id, "scenario_id": self.scenario_id,
                "parameter_path": self.parameter_path, "parameter_value": self.parameter_value,
                "metrics": self.metrics, "experiment_hash": self.experiment_hash}


class SensitivityAnalyzer:
    def __init__(self, runner: ExperimentRunner | None = None) -> None:
        self.runner = runner or ExperimentRunner()

    def sweep_policy_parameter(self, *, incident_id: str, created_at: str, seed: int,
                               evidence: Iterable[Event], candidates: Iterable[Candidate],
                               scenario: Scenario, policy_id: str, parameter: str,
                               values: Iterable[Any], trials: int = 3,
                               uncertainty_mode: str = "sample_candidates") -> list[SensitivityPoint]:
        evidence_values = list(evidence)
        candidate_values = list(candidates)
        points: list[SensitivityPoint] = []
        for value in values:
            policies = tuple(self._replace_parameter(policy, policy_id, parameter, value) for policy in scenario.policies)
            point_scenario = Scenario(f"{scenario.scenario_id}:{parameter}={value}", scenario.title, policies)
            plan = self.runner.build_plan(incident_id=incident_id, created_at=created_at, seed=seed,
                                          trials=trials, uncertainty_mode=uncertainty_mode,
                                          scenarios=(point_scenario,),
                                          parameters={"sensitivity": {"policy_id": policy_id,
                                                                       "parameter": parameter, "value": value}})
            result = self.runner.run(plan, evidence_values, candidate_values)
            metrics = result.scenario_summaries[point_scenario.scenario_id]
            identity = {"scenario_id": point_scenario.scenario_id, "parameter": parameter,
                        "value": value, "experiment_hash": result.experiment_hash}
            points.append(SensitivityPoint(deterministic_id("sensitivity", identity), point_scenario.scenario_id,
                                           f"{policy_id}.parameters.{parameter}", value,
                                           metrics, result.experiment_hash))
        return points

    def threshold_breakpoints(self, points: Iterable[SensitivityPoint], metric: str,
                              *, maximum: float | None = None, minimum: float | None = None) -> list[dict]:
        ordered = list(points)
        result: list[dict] = []
        previous_ok: bool | None = None
        for point in ordered:
            metric_value = float(point.metrics[metric]["mean"])
            ok = True
            if maximum is not None:
                ok &= metric_value <= maximum
            if minimum is not None:
                ok &= metric_value >= minimum
            if previous_ok is None or ok != previous_ok:
                result.append({"point_id": point.point_id, "parameter_value": point.parameter_value,
                               "metric": metric, "metric_value": metric_value, "constraint_satisfied": ok,
                               "transition": "initial" if previous_ok is None else "changed"})
            previous_ok = ok
        return result

    def monotonicity(self, points: Iterable[SensitivityPoint], metric: str) -> dict:
        values = [(point.parameter_value, float(point.metrics[metric]["mean"])) for point in points]
        increasing = all(values[index][1] <= values[index + 1][1] for index in range(len(values) - 1))
        decreasing = all(values[index][1] >= values[index + 1][1] for index in range(len(values) - 1))
        return {"metric": metric, "increasing": increasing, "decreasing": decreasing,
                "constant": increasing and decreasing, "values": values,
                "hash": digest_json(values)}

    def _replace_parameter(self, policy: PolicySpec, target_id: str, parameter: str, value: Any) -> PolicySpec:
        if policy.policy_id != target_id:
            return policy
        parameters = dict(policy.parameters)
        parameters[parameter] = value
        return PolicySpec(policy.policy_id, policy.kind, policy.priority, policy.enabled, parameters)


def aggregate_sensitivity(points: Iterable[SensitivityPoint], metric: str) -> dict:
    values = list(points)
    if not values:
        return {"metric": metric, "points": 0, "best": None, "worst": None}
    ranked = sorted(values, key=lambda point: (float(point.metrics[metric]["mean"]), str(point.parameter_value)))
    return {"metric": metric, "points": len(ranked),
            "best": {"parameter_value": ranked[0].parameter_value,
                     "mean": ranked[0].metrics[metric]["mean"], "point_id": ranked[0].point_id},
            "worst": {"parameter_value": ranked[-1].parameter_value,
                      "mean": ranked[-1].metrics[metric]["mean"], "point_id": ranked[-1].point_id}}
