from __future__ import annotations

import json
import unittest

from replay_lab.audit import AuditQueries
from replay_lab.errors import CheckpointError, IntegrityError
from replay_lab.journal import Journal
from replay_lab.provenance import ProvenanceLedger, fingerprint_package
from replay_lab.recovery import RecoveryPlanner
from replay_lab.replay import ReplayEngine

from tests.support import TempWorkspace, clean_events, event, incident_events, timestamp


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.events = clean_events()
        self.engine = ReplayEngine()
        self.planner = RecoveryPlanner(self.engine)

    def test_create_verified_checkpoint(self):
        checkpoint = self.planner.create_verified_checkpoint(self.events[:8], "cp", self.events[7].event_time)
        self.assertEqual(checkpoint.checkpoint_id, "cp")
        self.engine.verify_checkpoint(checkpoint)

    def test_select_resume_point(self):
        first = self.planner.create_verified_checkpoint(self.events[:2], "cp1", self.events[1].event_time)
        second = self.planner.create_verified_checkpoint(self.events[:5], "cp2", self.events[4].event_time)
        selected = self.planner.select_resume_point([second, first], self.events[6].event_time)
        self.assertEqual(selected.checkpoint_id, "cp2")

    def test_select_resume_point_can_return_none(self):
        checkpoint = self.planner.create_verified_checkpoint(self.events[:2], "cp", self.events[1].event_time)
        self.assertIsNone(self.planner.select_resume_point([checkpoint], timestamp(0)))

    def test_segments_cover_all_events(self):
        first = self.planner.create_verified_checkpoint(self.events[:4], "cp1", self.events[3].event_time)
        second = self.planner.create_verified_checkpoint(self.events[:9], "cp2", self.events[8].event_time)
        segments = self.planner.segment(self.events, [first, second])
        ids = [event_id for segment in segments for event_id in segment.event_ids]
        self.assertEqual(ids, [event.event_id for event in self.events])

    def test_segments_reject_duplicate_checkpoint(self):
        checkpoint = self.planner.create_verified_checkpoint(self.events[:2], "cp", self.events[1].event_time)
        with self.assertRaises(CheckpointError):
            self.planner.segment(self.events, [checkpoint, checkpoint])

    def test_repair_checkpoint_metadata_reports_missing_anchor(self):
        checkpoint = self.planner.create_verified_checkpoint(self.events[:2], "cp", self.events[1].event_time)
        changed = checkpoint.__class__(checkpoint.checkpoint_id, checkpoint.incident_id, checkpoint.event_time,
                                       ("missing",), checkpoint.state, checkpoint.state_hash, checkpoint.source)
        result = self.planner.repair_checkpoint_metadata(changed, self.events)
        self.assertEqual(result["missing_anchor_ids"], ["missing"])

    def test_verify_resume_state_equivalence_for_checkpoint_prefix(self):
        checkpoint = self.planner.create_verified_checkpoint(self.events[:8], "cp", self.events[7].event_time)
        result = self.planner.verify_resume(self.events, checkpoint)
        self.assertTrue(result.equivalent)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TempWorkspace()
        self.journal = Journal(self.temp.path / "audit.sqlite3")
        self.journal.register_incident("inc-001", "Synthetic", timestamp(0), {"synthetic": True})
        self.events = incident_events()
        self.journal.ingest_events(self.events + [self.events[0]])
        self.audit = AuditQueries(self.journal)

    def tearDown(self):
        self.journal.close()
        self.temp.close()

    def test_evidence_inventory(self):
        inventory = self.audit.evidence_inventory("inc-001")
        self.assertEqual(inventory["event_count"], len(self.events))
        self.assertEqual(inventory["agents"]["agent-alpha"], len(self.events))

    def test_duplicate_inventory(self):
        inventory = self.audit.duplicate_inventory("inc-001")
        self.assertEqual(inventory["exact"], 1)

    def test_rejection_inventory_empty(self):
        inventory = self.audit.rejection_inventory("inc-001")
        self.assertEqual(inventory["reasons"], [])

    def test_causal_neighborhood(self):
        neighborhood = self.audit.causal_neighborhood("inc-001", "e07", depth=1)
        ids = {item["event"]["event_id"] for item in neighborhood["nodes"]}
        self.assertIn("e06", ids)
        self.assertIn("e08", ids)

    def test_causal_neighborhood_missing_event(self):
        result = self.audit.causal_neighborhood("inc-001", "missing")
        self.assertFalse(result["found"])

    def test_application_exposure(self):
        exposure = self.audit.application_exposure("inc-001")
        wallet = next(item for item in exposure if item["application"] == "wallet")
        self.assertGreater(wallet["loss_proxy"], 0)
        self.assertIn("transfer", wallet["tools"])

    def test_snapshot_inventory_validates_hash(self):
        state = {"incident_id": "inc-001"}
        self.journal.save_snapshot("s", "inc-001", None, 0, None, state)
        inventory = self.audit.snapshot_inventory("inc-001")
        self.assertTrue(inventory[0]["state_hash_valid"])


class ProvenanceTests(unittest.TestCase):
    def test_add_and_export_round_trip(self):
        ledger = ProvenanceLedger()
        source = ledger.add_json("fixture", "fixture.json", {"a": 1})
        report = ledger.add_text("report", "report.txt", "ok", parent_hashes=[source.content_hash])
        exported = ledger.export()
        restored = ProvenanceLedger.from_export(exported)
        self.assertEqual(restored.export(), exported)
        self.assertEqual(restored.lineage(report.record_id)[0]["record_id"], report.record_id)

    def test_export_hash_mismatch_rejected(self):
        ledger = ProvenanceLedger()
        ledger.add_json("fixture", "fixture.json", {"a": 1})
        exported = ledger.export()
        exported["ledger_hash"] = "bad"
        with self.assertRaises(IntegrityError):
            ProvenanceLedger.from_export(exported)

    def test_verify_file(self):
        temp = TempWorkspace()
        try:
            path = temp.path / "x.txt"
            path.write_text("hello", encoding="utf-8")
            ledger = ProvenanceLedger()
            record = ledger.add_file("text", path)
            self.assertTrue(ledger.verify_file(record.record_id, path)["ok"])
            path.write_text("changed", encoding="utf-8")
            self.assertFalse(ledger.verify_file(record.record_id, path)["ok"])
        finally:
            temp.close()

    def test_missing_file_rejected(self):
        ledger = ProvenanceLedger()
        with self.assertRaises(IntegrityError):
            ledger.add_file("text", "/definitely/missing/file")

    def test_external_parent_reported(self):
        ledger = ProvenanceLedger()
        ledger.add_text("report", "r", "x", parent_hashes=["f" * 64])
        result = ledger.validate_graph()
        self.assertIn("f" * 64, result["external_parent_hashes"])

    def test_fingerprint_package(self):
        temp = TempWorkspace()
        try:
            (temp.path / "a.py").write_text("x=1\n", encoding="utf-8")
            (temp.path / "b.bin").write_bytes(b"ignored")
            result = fingerprint_package(temp.path)
            self.assertEqual([item["path"] for item in result["files"]], ["a.py"])
        finally:
            temp.close()


if __name__ == "__main__":
    unittest.main()
