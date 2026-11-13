from __future__ import annotations

import json
import sqlite3
import unittest

from replay_lab.canonical import (
    atomic_write_json, canonical_json, deterministic_id, digest_json, format_time,
    from_micros, mean, micros, parse_time, percentile, stable_unique,
)
from replay_lab.errors import ContractError, IntegrityError
from replay_lab.journal import Journal
from replay_lab.models import Checkpoint, Event, PolicySpec, ReplayState, event_index

from tests.support import TempWorkspace, checkpoint_from_state, event, event_dict, incident_events, timestamp


class CanonicalTests(unittest.TestCase):
    def test_canonical_json_sorts_keys_and_is_compact(self):
        self.assertEqual(canonical_json({"b": 1, "a": [2, 3]}), '{"a":[2,3],"b":1}')

    def test_digest_json_is_order_independent_for_objects(self):
        self.assertEqual(digest_json({"a": 1, "b": 2}), digest_json({"b": 2, "a": 1}))

    def test_parse_time_requires_timezone(self):
        with self.assertRaises(ContractError):
            parse_time("2026-11-12T02:00:00")

    def test_parse_time_normalizes_z(self):
        parsed = parse_time("2026-11-12T02:00:00Z")
        self.assertEqual(format_time(parsed), "2026-11-12T02:00:00.000000Z")

    def test_micros_round_trip(self):
        value = timestamp(12, 345)
        self.assertEqual(from_micros(micros(value)), value)

    def test_stable_unique_retains_first_order(self):
        self.assertEqual(stable_unique(["b", "a", "b", "c", "a"]), ["b", "a", "c"])

    def test_percentiles_interpolate(self):
        self.assertEqual(percentile([0, 10], 0.5), 5)
        self.assertEqual(percentile([1, 3, 9], 0.5), 3)
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_deterministic_id_stable(self):
        left = deterministic_id("x", {"a": 1})
        right = deterministic_id("x", {"a": 1})
        self.assertEqual(left, right)
        self.assertTrue(left.startswith("x_"))

    def test_atomic_write_json_replaces_file(self):
        temp = TempWorkspace()
        try:
            path = temp.path / "value.json"
            atomic_write_json(path, {"a": 1})
            atomic_write_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 2})
            self.assertFalse((temp.path / "value.json.tmp").exists())
        finally:
            temp.close()


class EventContractTests(unittest.TestCase):
    def test_event_hash_is_added(self):
        value = event("e1", 1, "agent_started", 0)
        self.assertEqual(value.evidence_hash, value.calculate_evidence_hash())

    def test_event_hash_mismatch_rejected(self):
        raw = event_dict("e1", 1, "agent_started", 0)
        raw["evidence_hash"] = "0" * 64
        with self.assertRaises(ContractError):
            Event.from_dict(raw)

    def test_unknown_event_kind_rejected(self):
        raw = event_dict("e1", 1, "agent_started", 0)
        raw["kind"] = "unknown"
        with self.assertRaises(ContractError):
            Event.from_dict(raw)

    def test_negative_duration_rejected(self):
        raw = event_dict("e1", 1, "agent_started", 0)
        raw["duration_ms"] = -1
        with self.assertRaises(ContractError):
            Event.from_dict(raw)

    def test_confidence_above_one_rejected(self):
        raw = event_dict("e1", 1, "agent_started", 0)
        raw["confidence"] = 1.01
        with self.assertRaises(ContractError):
            Event.from_dict(raw)

    def test_tags_are_sorted_and_unique(self):
        raw = event_dict("e1", 1, "agent_started", 0, tags=["b", "a", "b"])
        value = Event.from_dict(raw)
        self.assertEqual(value.tags, ("a", "b"))

    def test_event_index_accepts_exact_duplicates(self):
        value = event("e1", 1, "agent_started", 0)
        self.assertEqual(event_index([value, value]), {"e1": value})

    def test_event_index_rejects_conflicting_ids(self):
        left = event("e1", 1, "agent_started", 0)
        right = event("e1", 2, "agent_stopped", 1)
        with self.assertRaises(ContractError):
            event_index([left, right])

    def test_policy_spec_defaults(self):
        spec = PolicySpec.from_dict({"policy_id": "p", "kind": "call_block"})
        self.assertEqual(spec.priority, 100)
        self.assertTrue(spec.enabled)
        self.assertEqual(spec.parameters, {})

    def test_checkpoint_hash_validation(self):
        state = ReplayState("inc-001").to_dict()
        raw = {"checkpoint_id": "cp", "incident_id": "inc-001", "event_time": timestamp(1),
               "last_event_ids": [], "state": state, "state_hash": "bad"}
        with self.assertRaises(ContractError):
            Checkpoint.from_dict(raw)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = TempWorkspace()
        self.journal = Journal(self.temp.path / "lab.sqlite3")
        self.journal.register_incident("inc-001", "Synthetic", timestamp(0), {"synthetic": True})

    def tearDown(self):
        self.journal.close()
        self.temp.close()

    def test_register_incident_is_idempotent(self):
        self.journal.register_incident("inc-001", "Synthetic", timestamp(0), {"synthetic": True})
        row = self.journal.incident("inc-001")
        self.assertEqual(row["title"], "Synthetic")

    def test_register_incident_conflict_rejected(self):
        with self.assertRaises(IntegrityError):
            self.journal.register_incident("inc-001", "Changed", timestamp(0), {"synthetic": True})

    def test_ingest_and_event_order(self):
        values = incident_events()[:4]
        result = self.journal.ingest_events([values[2], values[0], values[3], values[1]])
        self.assertEqual(result["accepted"], 4)
        ordered = self.journal.load_events("inc-001", order="event")
        self.assertEqual([item.event_id for item in ordered], ["e01", "e02", "e03", "e04"])

    def test_exact_duplicate_recorded_not_reinserted(self):
        value = event("e1", 1, "agent_started", 0)
        first = self.journal.ingest_events([value])
        second = self.journal.ingest_events([value])
        self.assertEqual(first["accepted"], 1)
        self.assertEqual(second["duplicates"], 1)
        self.assertEqual(self.journal.counts("inc-001")["events"], 1)
        self.assertEqual(self.journal.counts("inc-001")["duplicates"], 1)

    def test_conflicting_event_id_rejected(self):
        left = event("e1", 1, "agent_started", 0)
        right = event("e1", 2, "agent_stopped", 1)
        self.journal.ingest_events([left])
        result = self.journal.ingest_events([right])
        self.assertEqual(result["conflicts"], 1)
        self.assertEqual(self.journal.counts("inc-001")["rejections"], 1)

    def test_source_sequence_conflict_rejected(self):
        left = event("e1", 1, "agent_started", 0)
        right = event("e2", 1, "agent_stopped", 1)
        self.journal.ingest_events([left])
        result = self.journal.ingest_events([right])
        self.assertEqual(result["conflicts"], 1)

    def test_load_arrival_order_preserved(self):
        values = [event("e3", 3, "task_completed", 3), event("e1", 1, "agent_started", 1),
                  event("e2", 2, "task_received", 2)]
        self.journal.ingest_events(values)
        self.assertEqual([item.event_id for item in self.journal.load_events("inc-001", order="arrival")],
                         ["e3", "e1", "e2"])

    def test_unknown_order_rejected(self):
        with self.assertRaises(IntegrityError):
            self.journal.load_events("inc-001", order="random")

    def test_checkpoint_round_trip(self):
        state = ReplayState("inc-001")
        state.active_agents.add("a")
        checkpoint = checkpoint_from_state("cp-1", state, 1, ("e1",))
        self.journal.save_checkpoint(checkpoint)
        loaded = self.journal.latest_checkpoint("inc-001")
        self.assertEqual(loaded.checkpoint_id, "cp-1")
        self.assertEqual(loaded.state_hash, checkpoint.state_hash)

    def test_checkpoint_identifier_conflict_rejected(self):
        first = checkpoint_from_state("cp", ReplayState("inc-001"), 1, ())
        changed = ReplayState("inc-001")
        changed.active_agents.add("x")
        second = checkpoint_from_state("cp", changed, 1, ())
        self.journal.save_checkpoint(first)
        with self.assertRaises(IntegrityError):
            self.journal.save_checkpoint(second)

    def test_snapshot_hash_verification(self):
        state = ReplayState("inc-001").to_dict()
        hashed = self.journal.save_snapshot("s1", "inc-001", None, 0, None, state)
        self.assertEqual(hashed, digest_json(state))
        self.assertEqual(self.journal.load_snapshot("s1"), state)

    def test_artifact_manifest_round_trip(self):
        self.journal.record_artifact("report", "report.txt", "a" * 64, 10, timestamp(2))
        manifest = self.journal.artifact_manifest()
        self.assertEqual(manifest[0]["artifact_name"], "report")
        self.assertEqual(manifest[0]["bytes"], 10)

    def test_integrity_passes_for_valid_rows(self):
        self.journal.ingest_events(incident_events()[:3])
        result = self.journal.verify_integrity("inc-001")
        self.assertTrue(result["ok"])
        self.assertEqual(result["counts"]["events"], 3)

    def test_database_foreign_keys_enabled(self):
        enabled = self.journal.connection.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(enabled, 1)


if __name__ == "__main__":
    unittest.main()
