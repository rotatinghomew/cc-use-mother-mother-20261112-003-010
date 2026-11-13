"""Verifiable timeline and report artifact generation."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Iterable

from .canonical import atomic_write_json, canonical_json, digest_bytes, digest_json, digest_text, format_time, parse_time
from .comparison import ComparisonReport
from .errors import VerificationError
from .experiments import ExperimentResult
from .models import Candidate, Event, Gap


class ArtifactWriter:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def write_all(self, *, incident: dict, evidence: Iterable[Event], gaps: Iterable[Gap],
                  candidates: Iterable[Candidate], causal_report: dict,
                  experiment: ExperimentResult, comparison: ComparisonReport) -> dict:
        evidence_values = list(evidence)
        gap_values = list(gaps)
        candidate_values = list(candidates)
        files: dict[str, dict] = {}
        files["evidence"] = self._write_json("evidence.json", [event.to_dict() for event in evidence_values])
        files["causality"] = self._write_json("causality.json", causal_report)
        files["gaps"] = self._write_json("gaps.json", [gap.to_dict() for gap in gap_values])
        files["candidates"] = self._write_json("interpolation-candidates.json",
                                               [candidate.to_dict() for candidate in candidate_values])
        files["experiment"] = self._write_json("experiment.json", experiment.to_dict())
        files["comparison"] = self._write_json("comparison.json", comparison.to_dict())
        timeline_rows = self._timeline_rows(experiment)
        files["timeline_json"] = self._write_json("timeline.json", timeline_rows)
        files["timeline_csv"] = self._write_text("timeline.csv", self._timeline_csv(timeline_rows))
        files["report"] = self._write_text("comparison-report.txt", self._report_text(incident, experiment, comparison))
        manifest = {
            "schema_version": 1,
            "incident_id": incident["incident_id"],
            "incident_metadata_hash": incident["metadata_hash"],
            "experiment_hash": experiment.experiment_hash,
            "comparison_hash": comparison.comparison_hash,
            "files": files,
        }
        chain = self._hash_chain(timeline_rows)
        manifest["timeline_chain"] = chain
        manifest["manifest_hash"] = digest_json(manifest)
        self._write_json("manifest.json", manifest)
        return manifest

    def verify(self) -> dict:
        manifest_path = self.workspace / "manifest.json"
        if not manifest_path.exists():
            raise VerificationError("manifest missing", details={"workspace": str(self.workspace)})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest_hash = manifest.pop("manifest_hash", None)
        actual_manifest_hash = digest_json(manifest)
        manifest["manifest_hash"] = expected_manifest_hash
        errors: list[dict] = []
        if expected_manifest_hash != actual_manifest_hash:
            errors.append({"type": "manifest_hash", "expected": expected_manifest_hash, "actual": actual_manifest_hash})
        for name, entry in manifest.get("files", {}).items():
            path = self.workspace / entry["path"]
            if not path.is_file():
                errors.append({"type": "missing_file", "name": name, "path": entry["path"]})
                continue
            content = path.read_bytes()
            actual = digest_bytes(content)
            if actual != entry["sha256"]:
                errors.append({"type": "file_hash", "name": name, "expected": entry["sha256"], "actual": actual})
            if len(content) != entry["bytes"]:
                errors.append({"type": "file_size", "name": name, "expected": entry["bytes"], "actual": len(content)})
        timeline_path = self.workspace / manifest["files"]["timeline_json"]["path"]
        if timeline_path.is_file():
            rows = json.loads(timeline_path.read_text(encoding="utf-8"))
            actual_chain = self._hash_chain(rows)
            if actual_chain != manifest.get("timeline_chain"):
                errors.append({"type": "timeline_chain", "expected": manifest.get("timeline_chain"), "actual": actual_chain})
        return {"ok": not errors, "errors": errors, "manifest": manifest}

    def _write_json(self, name: str, value: object) -> dict:
        path = self.workspace / name
        atomic_write_json(path, value)
        return self._entry(path)

    def _write_text(self, name: str, value: str) -> dict:
        path = self.workspace / name
        path.write_text(value, encoding="utf-8")
        return self._entry(path)

    def _entry(self, path: Path) -> dict:
        content = path.read_bytes()
        return {"path": path.name, "sha256": digest_bytes(content), "bytes": len(content)}

    def _timeline_rows(self, experiment: ExperimentResult) -> list[dict]:
        rows: list[dict] = []
        for trial in experiment.trials:
            for sequence, item in enumerate(trial.replay.state.timeline):
                row = dict(item)
                row.update({"experiment_id": experiment.plan.experiment_id, "trial_id": trial.trial_id,
                            "trial_index": trial.trial_index, "trial_seed": trial.trial_seed,
                            "scenario_id": trial.scenario_id, "sequence": sequence,
                            "selected_candidate_ids": list(trial.selected_candidate_ids)})
                rows.append(row)
        rows.sort(key=lambda item: (item["trial_index"], item["scenario_id"], item["sequence"], item["event_id"]))
        previous = "0" * 64
        for row in rows:
            material = dict(row)
            material["previous_hash"] = previous
            row_hash = digest_json(material)
            row["previous_hash"] = previous
            row["row_hash"] = row_hash
            previous = row_hash
        return rows

    def _timeline_csv(self, rows: list[dict]) -> str:
        fields = ["trial_index", "trial_id", "scenario_id", "sequence", "event_id", "event_time",
                  "observed_time", "agent_id", "kind", "application", "tool", "resource", "synthetic",
                  "confidence", "decision_action", "decision_allowed", "policy_ids", "delay_ms",
                  "cumulative_loss", "cumulative_availability_cost", "cumulative_delay_ms",
                  "previous_hash", "row_hash"]
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            decision = row["decision"]
            writer.writerow({
                "trial_index": row["trial_index"], "trial_id": row["trial_id"],
                "scenario_id": row["scenario_id"], "sequence": row["sequence"],
                "event_id": row["event_id"], "event_time": row["event_time"],
                "observed_time": row["observed_time"], "agent_id": row["agent_id"],
                "kind": row["kind"], "application": row.get("application") or "",
                "tool": row.get("tool") or "", "resource": row.get("resource") or "",
                "synthetic": int(bool(row["synthetic"])), "confidence": row["confidence"],
                "decision_action": decision["action"], "decision_allowed": int(bool(decision["allowed"])),
                "policy_ids": "|".join(decision["policy_ids"]), "delay_ms": decision["delay_ms"],
                "cumulative_loss": row["cumulative_loss"],
                "cumulative_availability_cost": row["cumulative_availability_cost"],
                "cumulative_delay_ms": row["cumulative_delay_ms"],
                "previous_hash": row["previous_hash"], "row_hash": row["row_hash"],
            })
        return stream.getvalue()

    def _report_text(self, incident: dict, experiment: ExperimentResult, comparison: ComparisonReport) -> str:
        lines = [
            "智能体事故事件重放与反事实遏制实验室", "",
            f"事故编号: {incident['incident_id']}", f"事故标题: {incident['title']}",
            f"实验编号: {experiment.plan.experiment_id}", f"随机种子: {experiment.plan.seed}",
            f"不确定性模式: {experiment.plan.uncertainty_mode}", f"试验次数: {experiment.plan.trials}",
            f"实验哈希: {experiment.experiment_hash}", f"比较哈希: {comparison.comparison_hash}", "",
            "策略比较（均值）", "",
        ]
        row_by_id = {row["scenario_id"]: row for row in comparison.rows}
        for rank, scenario_id in enumerate(comparison.rankings, 1):
            summary = experiment.scenario_summaries[scenario_id]
            row = row_by_id[scenario_id]
            lines.extend([
                f"{rank}. {scenario_id} - {summary['title']}",
                f"   损失代理: {summary['loss_proxy']['mean']:.3f}",
                f"   可用性代价: {summary['availability_cost']['mean']:.3f}",
                f"   响应时延(ms): {summary['response_delay_ms']['mean']:.3f}",
                f"   影响智能体/应用/资源: {summary['impact_agents']['mean']:.2f}/"
                f"{summary['impact_applications']['mean']:.2f}/{summary['impact_resources']['mean']:.2f}",
                f"   阻断事件: {summary['blocked_events']['mean']:.2f}",
                f"   相对基线损失减少: {row['loss_reduction']:.3f}",
                f"   可用性增量: {row['availability_increase']:.3f}",
                f"   加权得分: {row['weighted_score']:.6f}", "",
            ])
        lines.append("Pareto 前沿: " + ", ".join(comparison.pareto_front))
        lines.append("说明: 插值事件仅用于实验覆盖层，原始证据文件与 SQLite evidence_events 表保持不变。")
        lines.append("说明: 本报告是离线工程模拟，不替代监管审批、法律责任认定或真实终端处置。")
        return "\n".join(lines) + "\n"

    def _hash_chain(self, rows: list[dict]) -> dict:
        previous = "0" * 64
        for row in rows:
            material = dict(row)
            material.pop("row_hash", None)
            material["previous_hash"] = previous
            previous = digest_json(material)
        return {"algorithm": "sha256", "rows": len(rows), "root": previous}
