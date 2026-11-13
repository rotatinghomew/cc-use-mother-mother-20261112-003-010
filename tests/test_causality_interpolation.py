from __future__ import annotations

import unittest

from replay_lab.causality import CausalityAnalyzer
from replay_lab.diagnostics import Diagnostics
from replay_lab.errors import CausalityError
from replay_lab.interpolation import Interpolator
from replay_lab.models import Candidate, Event, Gap

from tests.support import clean_events, event, incident_events, timestamp


class CausalityTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = CausalityAnalyzer(clock_skew_ms=1000, gap_threshold_ms=8000)

    def test_orders_out_of_order_events_by_event_time(self):
        values = incident_events()
        report = self.analyzer.analyze([values[5], values[0], values[2], values[1]])
        self.assertEqual([item.event_id for item in report.ordered_events], ["e01", "e02", "e03", "e06"])

    def test_detects_missing_sequence_gap(self):
        report = self.analyzer.analyze(incident_events())
        gaps = [gap for gap in report.gaps if gap.before_event_id == "e09" and gap.after_event_id == "e11"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].missing_sequences, (10,))
        self.assertIn("sequence_discontinuity", gaps[0].reasons)

    def test_detects_silence_gap(self):
        report = self.analyzer.analyze(incident_events())
        gap = next(gap for gap in report.gaps if gap.before_event_id == "e09")
        self.assertIn("event_time_silence", gap.reasons)

    def test_detects_missing_parent(self):
        report = self.analyzer.analyze(incident_events())
        warnings = [item for item in report.warnings if item["type"] == "missing_parent"]
        self.assertTrue(any(item["parent_id"] == "e10" for item in warnings))
        self.assertTrue(any("missing_parent" in gap.reasons for gap in report.gaps))

    def test_rejects_parent_after_child(self):
        child = event("c", 1, "task_received", 1, parent_ids=["p"])
        parent = event("p", 2, "agent_started", 10)
        report = self.analyzer.analyze([child, parent])
        self.assertTrue(any(item["type"] == "parent_after_child" for item in report.errors))

    def test_detects_self_parent(self):
        value = event("e", 1, "task_received", 1, parent_ids=["e"])
        report = self.analyzer.analyze([value])
        self.assertTrue(any(item["type"] == "self_parent" for item in report.errors))

    def test_detects_cycle(self):
        left = event("a", 1, "task_received", 1, parent_ids=["b"])
        right = event("b", 2, "task_completed", 2, parent_ids=["a"])
        report = self.analyzer.analyze([left, right])
        self.assertTrue(any(item["type"] == "causal_cycle" for item in report.errors))

    def test_strict_mode_raises(self):
        left = event("a", 1, "task_received", 1, parent_ids=["b"])
        right = event("b", 2, "task_completed", 2, parent_ids=["a"])
        with self.assertRaises(CausalityError):
            self.analyzer.analyze([left, right], strict=True)

    def test_clock_skew_warning(self):
        value = event("a", 1, "agent_started", 10, observed_seconds=1)
        report = self.analyzer.analyze([value])
        self.assertTrue(any(item["type"] == "observed_before_event" for item in report.warnings))

    def test_conflicting_terminal_outcomes(self):
        left = event("a", 1, "tool_call_completed", 1, correlation_id="c", outcome="ok")
        right = event("b", 2, "tool_call_failed", 2, correlation_id="c", outcome="failed")
        report = self.analyzer.analyze([left, right])
        self.assertTrue(any(item["type"] == "conflicting_terminal_outcomes" for item in report.errors))

    def test_roots_and_leaves(self):
        values = clean_events()[:4]
        report = self.analyzer.analyze(values)
        self.assertEqual(report.roots, ("e01",))
        self.assertEqual(report.leaves, ("e04",))

    def test_critical_path_uses_duration(self):
        values = [event("a", 1, "task_received", 1, duration_ms=10),
                  event("b", 2, "tool_call_started", 2, parent_ids=["a"], duration_ms=30),
                  event("c", 3, "task_completed", 3, parent_ids=["a"], duration_ms=5)]
        report = self.analyzer.analyze(values)
        self.assertEqual(self.analyzer.critical_path(report), ["a", "b"])

    def test_event_time_windows(self):
        report = self.analyzer.analyze(clean_events())
        windows = self.analyzer.event_time_windows(report, 5000)
        self.assertGreaterEqual(len(windows), 3)
        self.assertEqual(sum(item["count"] for item in windows), len(clean_events()))

    def test_event_time_window_width_must_be_positive(self):
        report = self.analyzer.analyze(clean_events())
        with self.assertRaises(CausalityError):
            self.analyzer.event_time_windows(report, 0)

    def test_complete_stream_has_no_sequence_gap_for_call_two(self):
        report = self.analyzer.analyze(clean_events())
        self.assertFalse(any(gap.before_event_id == "e09" and gap.after_event_id == "e11" for gap in report.gaps))


class InterpolationTests(unittest.TestCase):
    def setUp(self):
        self.events = incident_events()
        self.report = CausalityAnalyzer(gap_threshold_ms=8000).analyze(self.events)
        self.interpolator = Interpolator(max_candidates=4, confidence_floor=0.1)

    def test_generates_candidate_for_missing_tool_start(self):
        candidates = self.interpolator.generate(self.report.gaps, self.events)
        matching = [candidate for candidate in candidates
                    if any(event.kind == "tool_call_started" and event.correlation_id == "call-2" for event in candidate.events)]
        self.assertTrue(matching)

    def test_candidates_are_synthetic_non_evidence(self):
        candidates = self.interpolator.generate(self.report.gaps, self.events)
        generated = [event for candidate in candidates for event in candidate.events]
        self.assertTrue(generated)
        self.assertTrue(all(event.synthetic for event in generated))
        self.assertTrue(all("non_evidence" in event.tags for event in generated))

    def test_candidate_confidence_bounded(self):
        candidates = self.interpolator.generate(self.report.gaps, self.events)
        self.assertTrue(all(0 <= item.confidence <= 1 for item in candidates))

    def test_overlay_does_not_mutate_evidence(self):
        before = [event.to_dict() for event in self.events]
        candidate = self.interpolator.generate(self.report.gaps, self.events)[0]
        overlay = self.interpolator.overlay(self.events, candidate)
        self.assertEqual(before, [event.to_dict() for event in self.events])
        self.assertGreater(len(overlay), len(self.events))

    def test_overlay_without_candidate_is_sorted_copy(self):
        overlay = self.interpolator.overlay(reversed(self.events), None)
        self.assertEqual(overlay[0].event_id, "e01")
        self.assertEqual(len(overlay), len(self.events))

    def test_confidence_floor_filters_candidates(self):
        strict = Interpolator(confidence_floor=0.9)
        self.assertEqual(strict.generate(self.report.gaps, self.events), [])

    def test_max_candidates_applies_per_gap(self):
        limited = Interpolator(max_candidates=1, confidence_floor=0.0)
        candidates = limited.generate(self.report.gaps, self.events)
        counts = {}
        for candidate in candidates:
            counts[candidate.gap_id] = counts.get(candidate.gap_id, 0) + 1
        self.assertTrue(all(value <= 1 for value in counts.values()))

    def test_protocol_transition_candidate(self):
        before = event("a", 1, "tool_call_requested", 1, correlation_id="c")
        after = event("b", 3, "tool_call_completed", 3, correlation_id="c", parent_ids=["a"])
        gap = Gap("g", "inc-001", "agent-alpha", before.event_time, after.event_time,
                  "a", "b", (2,), ("sequence_discontinuity",))
        candidates = self.interpolator.generate([gap], [before, after])
        self.assertEqual(candidates[0].events[0].kind, "tool_call_started")
        self.assertGreaterEqual(candidates[0].confidence, 0.8)

    def test_correlation_mismatch_reduces_confidence(self):
        before = event("a", 1, "tool_call_requested", 1, correlation_id="x")
        after = event("b", 3, "tool_call_completed", 3, correlation_id="y", parent_ids=["a"])
        gap = Gap("g", "inc-001", "agent-alpha", before.event_time, after.event_time,
                  "a", "b", (2,), ("sequence_discontinuity",))
        candidates = self.interpolator.generate([gap], [before, after])
        self.assertLess(candidates[0].confidence, 0.8)
        self.assertTrue(any("correlation_mismatch" in candidate.contradictions for candidate in candidates))

    def test_missing_parent_candidate_uses_parent_id(self):
        after = event("b", 3, "tool_call_completed", 3, correlation_id="c", parent_ids=["missing"])
        gap = Gap("g", "inc-001", "agent-alpha", after.event_time, after.event_time,
                  "missing", "b", (), ("missing_parent",))
        candidates = self.interpolator.generate([gap], [after])
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].events[0].parent_ids, ("missing",))

    def test_candidate_ids_are_deterministic(self):
        left = self.interpolator.generate(self.report.gaps, self.events)
        right = self.interpolator.generate(self.report.gaps, self.events)
        self.assertEqual([item.candidate_id for item in left], [item.candidate_id for item in right])


class DiagnosticsTests(unittest.TestCase):
    def test_fixture_findings_include_gap_and_risk(self):
        events = incident_events()
        report = CausalityAnalyzer(gap_threshold_ms=8000).analyze(events)
        candidates = Interpolator().generate(report.gaps, events)
        findings = Diagnostics().fixture_findings(events, report.gaps, candidates)
        codes = {item.code for item in findings}
        self.assertIn("missing_sequence", codes)
        self.assertIn("high_loss_mutation", codes)

    def test_evidence_preservation_detects_changes(self):
        events = incident_events()
        changed = list(events)
        changed[0] = event("e01", 1, "agent_started", 2)
        result = Diagnostics().evidence_preservation(events, changed)
        self.assertFalse(result["preserved"])
        self.assertEqual(result["changed_evidence_ids"], ["e01"])

    def test_evidence_preservation_ignores_synthetic_additions(self):
        events = incident_events()
        synthetic = event("hyp", 100, "tool_call_started", 10, synthetic=True, confidence=0.5)
        result = Diagnostics().evidence_preservation(events, events + [synthetic])
        self.assertTrue(result["preserved"])


if __name__ == "__main__":
    unittest.main()
