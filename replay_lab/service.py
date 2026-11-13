"""Application service joining journal, replay, experiments, and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, deterministic_id, digest_json, format_time, parse_time, read_json
from .causality import CausalityAnalyzer
from .comparison import Comparator
from .errors import ContractError, LabError, VerificationError
from .experiments import ExperimentRunner, scenarios_from_dict
from .interpolation import Interpolator
from .journal import Journal
from .models import Checkpoint, Event
from .replay import ReplayEngine
from .timeline import ArtifactWriter


class LaboratoryService:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.database_path = self.workspace / "laboratory.sqlite3"

    def run_fixture(self, fixture_path: str | Path) -> dict:
        fixture = read_json(Path(fixture_path))
        return self.run(fixture)

    def run(self, fixture: dict[str, Any]) -> dict:
        if not isinstance(fixture, dict):
            raise ContractError("fixture root must be an object")
        incident = self._incident_contract(fixture.get("incident"))
        raw_events = fixture.get("events")
        raw_checkpoints = fixture.get("checkpoints", [])
        raw_experiment = fixture.get("experiment")
        if not isinstance(raw_events, list) or not raw_events:
            raise ContractError("fixture requires non-empty events array")
        if not isinstance(raw_checkpoints, list):
            raise ContractError("checkpoints must be an array")
        if not isinstance(raw_experiment, dict):
            raise ContractError("fixture requires experiment object")
        with Journal(self.database_path) as journal:
            journal.register_incident(incident["incident_id"], incident["title"], incident["created_at"], incident["metadata"])
            parsed_events: list[Event] = []
            rejected = 0
            for raw in raw_events:
                try:
                    parsed_events.append(Event.from_dict(raw))
                except LabError as exc:
                    rejected += 1
                    journal.reject_raw(raw if isinstance(raw, dict) else {"raw": raw}, exc.code + ":" + exc.message)
            ingest = journal.ingest_events(parsed_events)
            checkpoints: list[Checkpoint] = []
            for raw in raw_checkpoints:
                checkpoint = Checkpoint.from_dict(raw)
                if checkpoint.incident_id != incident["incident_id"]:
                    raise ContractError("checkpoint incident mismatch", details={"checkpoint_id": checkpoint.checkpoint_id})
                journal.save_checkpoint(checkpoint)
                checkpoints.append(checkpoint)
            evidence = journal.load_events(incident["incident_id"], order="event")
            analyzer_config = fixture.get("analysis", {})
            if not isinstance(analyzer_config, dict):
                raise ContractError("analysis must be an object")
            analyzer = CausalityAnalyzer(clock_skew_ms=int(analyzer_config.get("clock_skew_ms", 2000)),
                                         gap_threshold_ms=int(analyzer_config.get("gap_threshold_ms", 10000)))
            causal = analyzer.analyze(evidence, strict=False)
            interpolator = Interpolator(max_candidates=int(analyzer_config.get("max_candidates_per_gap", 4)),
                                        confidence_floor=float(analyzer_config.get("confidence_floor", 0.15)))
            candidates = interpolator.generate(causal.gaps, evidence)
            journal.save_candidates(incident["incident_id"], candidates)
            scenarios = scenarios_from_dict(raw_experiment.get("scenarios", []))
            runner = ExperimentRunner(ReplayEngine(), interpolator)
            plan = runner.build_plan(
                incident_id=incident["incident_id"],
                created_at=str(raw_experiment.get("created_at", incident["created_at"])),
                seed=int(raw_experiment.get("seed", 0)),
                trials=int(raw_experiment.get("trials", 1)),
                uncertainty_mode=str(raw_experiment.get("uncertainty_mode", "evidence_only")),
                scenarios=scenarios,
                parameters=raw_experiment.get("parameters", {}),
                experiment_id=raw_experiment.get("experiment_id"),
            )
            journal.save_experiment(plan.experiment_id, plan.incident_id, plan.created_at, plan.seed, plan.to_dict())
            checkpoint = journal.latest_checkpoint(incident["incident_id"], raw_experiment.get("checkpoint_at_or_before"))
            experiment = runner.run(plan, evidence, candidates, checkpoint=checkpoint)
            for trial in experiment.trials:
                journal.save_run(trial.trial_id, plan.experiment_id, trial.scenario_id, plan.created_at, trial.to_dict())
                snapshot_id = deterministic_id("snapshot", {"trial_id": trial.trial_id, "state_hash": trial.replay.state_hash})
                journal.save_snapshot(snapshot_id, incident["incident_id"], trial.replay.run_id,
                                      len(trial.replay.processed_event_ids),
                                      trial.replay.processed_event_ids[-1] if trial.replay.processed_event_ids else None,
                                      trial.replay.state.to_dict())
            comparator = Comparator()
            baseline_id = str(raw_experiment.get("baseline_scenario", scenarios[0].scenario_id))
            comparison = comparator.compare(experiment, baseline_id=baseline_id,
                                            weights=raw_experiment.get("comparison_weights"))
            incident_row = journal.incident(incident["incident_id"])
            writer = ArtifactWriter(self.workspace)
            manifest = writer.write_all(incident=incident_row, evidence=evidence, gaps=causal.gaps,
                                        candidates=candidates, causal_report=causal.to_dict(),
                                        experiment=experiment, comparison=comparison)
            for name, entry in manifest["files"].items():
                journal.record_artifact(name, entry["path"], entry["sha256"], entry["bytes"], plan.created_at)
            integrity = journal.verify_integrity(incident["incident_id"])
            artifact_verification = writer.verify()
            summary = {
                "ok": integrity["ok"] and artifact_verification["ok"],
                "incident_id": incident["incident_id"],
                "database": self.database_path.name,
                "ingest": {**ingest, "parse_rejections": rejected},
                "causality": {"valid": causal.valid, "errors": len(causal.errors),
                              "warnings": len(causal.warnings), "gaps": len(causal.gaps),
                              "critical_path": analyzer.critical_path(causal)},
                "interpolation": {"candidates": len(candidates),
                                  "evidence_events_unchanged": journal.counts(incident["incident_id"])["events"] == len(evidence)},
                "experiment": {"experiment_id": plan.experiment_id, "trials": len(experiment.trials),
                               "experiment_hash": experiment.experiment_hash},
                "comparison": comparison.to_dict(),
                "integrity": integrity,
                "artifacts": artifact_verification,
            }
            atomic_write_json(self.workspace / "summary.json", summary)
            return summary

    def verify(self) -> dict:
        writer = ArtifactWriter(self.workspace)
        artifacts = writer.verify()
        database_ok = self.database_path.is_file()
        result = {"ok": artifacts["ok"] and database_ok, "database_exists": database_ok,
                  "artifacts": artifacts}
        if database_ok:
            with Journal(self.database_path) as journal:
                rows = journal.connection.execute("SELECT incident_id FROM incidents ORDER BY incident_id").fetchall()
                result["incidents"] = []
                for row in rows:
                    result["incidents"].append(journal.verify_integrity(row["incident_id"]))
                result["ok"] = result["ok"] and all(item["ok"] for item in result["incidents"])
        return result

    def _incident_contract(self, raw: Any) -> dict:
        if not isinstance(raw, dict):
            raise ContractError("fixture requires incident object")
        incident_id = str(raw.get("incident_id", "")).strip()
        title = str(raw.get("title", "")).strip()
        created_at = str(raw.get("created_at", "")).strip()
        metadata = raw.get("metadata", {})
        if not incident_id or not title or not created_at or not isinstance(metadata, dict):
            raise ContractError("incident requires id, title, created_at, and metadata")
        parse_time(created_at)
        return {"incident_id": incident_id, "title": title, "created_at": created_at, "metadata": metadata}
