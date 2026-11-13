from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from replay_lab.causality import CausalityAnalyzer
from replay_lab.comparison import Comparator
from replay_lab.experiments import ExperimentRunner, Scenario
from replay_lab.interpolation import Interpolator
from replay_lab.service import LaboratoryService
from replay_lab.timeline import ArtifactWriter

from tests.support import TempWorkspace, fixture_dict, incident_events, policy, timestamp


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TempWorkspace()
        self.workspace = self.temp.path / "workspace"

    def tearDown(self):
        self.temp.close()

    def test_run_fixture_creates_complete_workspace(self):
        fixture_path = self.temp.write_fixture()
        result = LaboratoryService(self.workspace).run_fixture(fixture_path)
        self.assertTrue(result["ok"])
        self.assertTrue((self.workspace / "laboratory.sqlite3").is_file())
        self.assertTrue((self.workspace / "manifest.json").is_file())
        self.assertTrue((self.workspace / "comparison-report.txt").is_file())

    def test_run_records_duplicate(self):
        result = LaboratoryService(self.workspace).run(fixture_dict())
        self.assertEqual(result["ingest"]["duplicates"], 1)
        self.assertGreaterEqual(result["causality"]["gaps"], 1)

    def test_run_preserves_evidence_count(self):
        result = LaboratoryService(self.workspace).run(fixture_dict())
        self.assertTrue(result["interpolation"]["evidence_events_unchanged"])
        self.assertGreater(result["interpolation"]["candidates"], 0)

    def test_verify_after_run(self):
        service = LaboratoryService(self.workspace)
        service.run(fixture_dict())
        result = service.verify()
        self.assertTrue(result["ok"])
        self.assertTrue(result["database_exists"])

    def test_verify_detects_tampered_artifact(self):
        service = LaboratoryService(self.workspace)
        service.run(fixture_dict())
        (self.workspace / "comparison-report.txt").write_text("tampered", encoding="utf-8")
        result = service.verify()
        self.assertFalse(result["ok"])
        self.assertTrue(any(item["type"] == "file_hash" for item in result["artifacts"]["errors"]))

    def test_repeated_run_is_idempotent(self):
        service = LaboratoryService(self.workspace)
        first = service.run(fixture_dict())
        second = service.run(fixture_dict())
        self.assertTrue(second["ok"])
        self.assertEqual(first["experiment"]["experiment_hash"], second["experiment"]["experiment_hash"])

    def test_missing_incident_rejected(self):
        fixture = fixture_dict()
        fixture.pop("incident")
        with self.assertRaises(Exception):
            LaboratoryService(self.workspace).run(fixture)

    def test_empty_events_rejected(self):
        fixture = fixture_dict()
        fixture["events"] = []
        with self.assertRaises(Exception):
            LaboratoryService(self.workspace).run(fixture)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = TempWorkspace()
        events = incident_events()
        causal = CausalityAnalyzer(gap_threshold_ms=8000).analyze(events)
        candidates = Interpolator().generate(causal.gaps, events)
        scenarios = (Scenario("baseline", "Baseline", ()),
                     Scenario("block", "Block", (policy("call_block", tools=["transfer"]),)))
        runner = ExperimentRunner()
        plan = runner.build_plan(incident_id="inc-001", created_at=timestamp(40), seed=5, trials=2,
                                 uncertainty_mode="best_candidate", scenarios=scenarios)
        self.experiment = runner.run(plan, events, candidates)
        self.comparison = Comparator().compare(self.experiment, baseline_id="baseline")
        self.causal = causal
        self.events = events
        self.candidates = candidates

    def tearDown(self):
        self.temp.close()

    def test_write_all_and_verify(self):
        writer = ArtifactWriter(self.temp.path)
        incident = {"incident_id": "inc-001", "title": "Synthetic", "metadata_hash": "a" * 64}
        manifest = writer.write_all(incident=incident, evidence=self.events, gaps=self.causal.gaps,
                                    candidates=self.candidates, causal_report=self.causal.to_dict(),
                                    experiment=self.experiment, comparison=self.comparison)
        self.assertEqual(manifest["timeline_chain"]["rows"],
                         sum(len(trial.replay.state.timeline) for trial in self.experiment.trials))
        self.assertTrue(writer.verify()["ok"])

    def test_timeline_rows_have_hash_chain(self):
        writer = ArtifactWriter(self.temp.path)
        rows = writer._timeline_rows(self.experiment)
        self.assertEqual(rows[0]["previous_hash"], "0" * 64)
        for index in range(1, len(rows)):
            self.assertEqual(rows[index]["previous_hash"], rows[index - 1]["row_hash"])

    def test_csv_contains_expected_headers(self):
        writer = ArtifactWriter(self.temp.path)
        text = writer._timeline_csv(writer._timeline_rows(self.experiment))
        header = text.splitlines()[0]
        self.assertIn("row_hash", header)
        self.assertIn("decision_action", header)

    def test_report_mentions_non_evidence_boundary(self):
        writer = ArtifactWriter(self.temp.path)
        incident = {"incident_id": "inc-001", "title": "Synthetic"}
        text = writer._report_text(incident, self.experiment, self.comparison)
        self.assertIn("原始证据", text)
        self.assertIn("不替代", text)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = TempWorkspace()
        self.fixture = self.temp.write_fixture()
        self.workspace = self.temp.path / "cli-workspace"
        self.project = Path(__file__).resolve().parents[1]

    def tearDown(self):
        self.temp.close()

    def command(self, *args):
        return subprocess.run([sys.executable, "-m", "replay_lab", *args], cwd=self.project,
                              capture_output=True, text=True, timeout=30)

    def test_cli_run(self):
        result = self.command("run", "--fixture", str(self.fixture), "--workspace", str(self.workspace))
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])

    def test_cli_verify(self):
        self.command("run", "--fixture", str(self.fixture), "--workspace", str(self.workspace))
        result = self.command("verify", "--workspace", str(self.workspace))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["ok"])

    def test_cli_inspect(self):
        self.command("run", "--fixture", str(self.fixture), "--workspace", str(self.workspace))
        result = self.command("inspect", "--workspace", str(self.workspace))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["incident_id"], "inc-001")

    def test_cli_missing_fixture_fails(self):
        result = self.command("run", "--fixture", str(self.temp.path / "missing.json"),
                              "--workspace", str(self.workspace))
        self.assertEqual(result.returncode, 2)
        self.assertFalse(json.loads(result.stderr)["ok"])

    def test_cli_verify_missing_workspace_fails(self):
        result = self.command("verify", "--workspace", str(self.workspace))
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
