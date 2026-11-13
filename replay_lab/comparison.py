"""Cross-scenario comparison, dominance checks, and trade-off ranking."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import digest_json
from .experiments import ExperimentResult


METRICS = ("loss_proxy", "availability_cost", "response_delay_ms", "impact_agents",
           "impact_applications", "impact_resources", "blocked_events")


@dataclass(frozen=True)
class ComparisonReport:
    baseline_id: str
    rows: tuple[dict, ...]
    pareto_front: tuple[str, ...]
    rankings: tuple[str, ...]
    comparison_hash: str

    def to_dict(self) -> dict:
        return {"baseline_id": self.baseline_id, "rows": list(self.rows),
                "pareto_front": list(self.pareto_front), "rankings": list(self.rankings),
                "comparison_hash": self.comparison_hash}


class Comparator:
    def compare(self, result: ExperimentResult, *, baseline_id: str | None = None,
                weights: dict[str, float] | None = None) -> ComparisonReport:
        summaries = result.scenario_summaries
        if not summaries:
            raise ValueError("experiment contains no scenario summaries")
        baseline_id = baseline_id or next(iter(summaries))
        if baseline_id not in summaries:
            raise ValueError(f"baseline scenario not found: {baseline_id}")
        baseline = summaries[baseline_id]
        weights = weights or {"loss_proxy": 0.45, "availability_cost": 0.2, "response_delay_ms": 0.1,
                              "impact_agents": 0.1, "impact_applications": 0.05,
                              "impact_resources": 0.1, "blocked_events": 0.0}
        rows: list[dict] = []
        for scenario_id, summary in summaries.items():
            row = {"scenario_id": scenario_id, "title": summary["title"]}
            score = 0.0
            for metric in METRICS:
                value = float(summary[metric]["mean"])
                base = float(baseline[metric]["mean"])
                delta = value - base
                improvement = base - value
                normalized = improvement / max(abs(base), abs(value), 1.0)
                row[metric] = {"mean": value, "baseline": base, "delta": delta,
                               "relative_change": delta / abs(base) if base else (0.0 if value == 0 else 1.0)}
                score += normalized * float(weights.get(metric, 0.0))
            loss_reduction = baseline["loss_proxy"]["mean"] - summary["loss_proxy"]["mean"]
            availability_increase = summary["availability_cost"]["mean"] - baseline["availability_cost"]["mean"]
            row["loss_reduction"] = loss_reduction
            row["availability_increase"] = availability_increase
            row["containment_efficiency"] = loss_reduction / max(availability_increase, 0.001)
            row["weighted_score"] = score
            rows.append(row)
        rows.sort(key=lambda item: (-item["weighted_score"], item["scenario_id"]))
        pareto = self._pareto(rows)
        rankings = tuple(row["scenario_id"] for row in rows)
        identity = {"baseline_id": baseline_id, "rows": rows, "pareto_front": pareto,
                    "rankings": rankings, "weights": weights}
        return ComparisonReport(baseline_id, tuple(rows), tuple(pareto), rankings, digest_json(identity))

    def _pareto(self, rows: list[dict]) -> list[str]:
        front: list[str] = []
        for candidate in rows:
            candidate_values = self._objective_values(candidate)
            dominated = False
            for other in rows:
                if other is candidate:
                    continue
                other_values = self._objective_values(other)
                no_worse = all(left <= right for left, right in zip(other_values, candidate_values))
                strictly_better = any(left < right for left, right in zip(other_values, candidate_values))
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate["scenario_id"])
        return sorted(front)

    def _objective_values(self, row: dict) -> tuple[float, ...]:
        return (row["loss_proxy"]["mean"], row["availability_cost"]["mean"],
                row["response_delay_ms"]["mean"], row["impact_agents"]["mean"],
                row["impact_resources"]["mean"])

    def verify(self, report: ComparisonReport, result: ExperimentResult) -> bool:
        recreated = self.compare(result, baseline_id=report.baseline_id)
        return recreated.comparison_hash == report.comparison_hash
