from __future__ import annotations

import unittest

from replay_lab.causality import CausalityAnalyzer
from replay_lab.diagnostics import Diagnostics
from replay_lab.experiments import Scenario
from replay_lab.interpolation import Interpolator
from replay_lab.sensitivity import SensitivityAnalyzer, aggregate_sensitivity

from tests.support import incident_events, policy, timestamp


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.events = incident_events()
        report = CausalityAnalyzer(gap_threshold_ms=8000).analyze(self.events)
        self.candidates = Interpolator().generate(report.gaps, self.events)
        self.scenario = Scenario("rate", "Rate", (policy("rate_limit", policy_id="rate-policy",
                                                         limit=1, window_ms=10000,
                                                         scope="agent", mode="delay"),))
        self.analyzer = SensitivityAnalyzer()

    def test_sweep_creates_point_per_value(self):
        points = self.analyzer.sweep_policy_parameter(
            incident_id="inc-001", created_at=timestamp(40), seed=2,
            evidence=self.events, candidates=self.candidates, scenario=self.scenario,
            policy_id="rate-policy", parameter="window_ms", values=[1000, 5000, 10000],
            trials=2, uncertainty_mode="evidence_only")
        self.assertEqual(len(points), 3)
        self.assertEqual([item.parameter_value for item in points], [1000, 5000, 10000])

    def test_sweep_is_deterministic(self):
        kwargs = dict(incident_id="inc-001", created_at=timestamp(40), seed=2,
                      evidence=self.events, candidates=self.candidates, scenario=self.scenario,
                      policy_id="rate-policy", parameter="window_ms", values=[1000, 5000],
                      trials=2, uncertainty_mode="evidence_only")
        left = self.analyzer.sweep_policy_parameter(**kwargs)
        right = self.analyzer.sweep_policy_parameter(**kwargs)
        self.assertEqual([item.point_id for item in left], [item.point_id for item in right])

    def test_monotonicity_has_hash(self):
        points = self.analyzer.sweep_policy_parameter(
            incident_id="inc-001", created_at=timestamp(40), seed=2,
            evidence=self.events, candidates=self.candidates, scenario=self.scenario,
            policy_id="rate-policy", parameter="window_ms", values=[1000, 5000, 10000],
            trials=1, uncertainty_mode="evidence_only")
        result = self.analyzer.monotonicity(points, "response_delay_ms")
        self.assertIn("hash", result)
        self.assertEqual(len(result["values"]), 3)

    def test_threshold_breakpoints(self):
        points = self.analyzer.sweep_policy_parameter(
            incident_id="inc-001", created_at=timestamp(40), seed=2,
            evidence=self.events, candidates=self.candidates, scenario=self.scenario,
            policy_id="rate-policy", parameter="window_ms", values=[1000, 5000, 10000],
            trials=1, uncertainty_mode="evidence_only")
        result = self.analyzer.threshold_breakpoints(points, "response_delay_ms", maximum=1)
        self.assertTrue(result)
        self.assertEqual(result[0]["transition"], "initial")

    def test_aggregate_sensitivity(self):
        points = self.analyzer.sweep_policy_parameter(
            incident_id="inc-001", created_at=timestamp(40), seed=2,
            evidence=self.events, candidates=self.candidates, scenario=self.scenario,
            policy_id="rate-policy", parameter="window_ms", values=[1000, 5000],
            trials=1, uncertainty_mode="evidence_only")
        result = aggregate_sensitivity(points, "response_delay_ms")
        self.assertEqual(result["points"], 2)
        self.assertIn("best", result)

    def test_aggregate_empty(self):
        self.assertEqual(aggregate_sensitivity([], "loss_proxy")["points"], 0)


class MoreDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.events = incident_events()
        self.diagnostics = Diagnostics()

    def test_policy_findings_warn_uncovered_permission(self):
        findings = self.diagnostics.policy_findings([], self.events)
        self.assertIn("risky_permission_uncovered", {item.code for item in findings})

    def test_policy_findings_detect_broad_block(self):
        findings = self.diagnostics.policy_findings([policy("call_block", policy_id="p")], self.events)
        self.assertIn("broad_call_block", {item.code for item in findings})

    def test_policy_findings_disabled_policy(self):
        spec = policy("call_block", policy_id="p")
        disabled = spec.__class__(spec.policy_id, spec.kind, spec.priority, False, spec.parameters)
        findings = self.diagnostics.policy_findings([disabled], self.events)
        self.assertIn("disabled_policy", {item.code for item in findings})

    def test_policy_findings_shared_priority(self):
        findings = self.diagnostics.policy_findings([
            policy("call_block", policy_id="a", priority=10, tools=["x"]),
            policy("call_block", policy_id="b", priority=10, tools=["y"]),
        ], self.events)
        self.assertIn("shared_priority", {item.code for item in findings})

    def test_acceptance_checks_scenarios(self):
        summary = {"x": {"loss_proxy": {"mean": 2}, "availability_cost": {"mean": 1},
                         "response_delay_ms": {"mean": 10}}}
        result = self.diagnostics.experiment_acceptance(summary, max_loss=3, max_availability=2, max_delay_ms=20)
        self.assertTrue(result["ok"])

    def test_acceptance_failure(self):
        summary = {"x": {"loss_proxy": {"mean": 5}, "availability_cost": {"mean": 1},
                         "response_delay_ms": {"mean": 10}}}
        result = self.diagnostics.experiment_acceptance(summary, max_loss=3)
        self.assertFalse(result["ok"])

    def test_coverage_matrix_counts_sources_agents_and_applications(self):
        result = self.diagnostics.coverage_matrix(self.events)
        self.assertEqual(result["event_count"], len(self.events))
        self.assertGreaterEqual(len(result["sources"]), 1)
        self.assertIn(self.events[0].agent_id, result["agents"])
        self.assertIn(self.events[0].application or "unattributed", result["applications"])
        self.assertEqual(len(result["coverage_hash"]), 64)

    def test_coverage_matrix_is_order_independent(self):
        left = self.diagnostics.coverage_matrix(self.events)
        right = self.diagnostics.coverage_matrix(reversed(self.events))
        self.assertEqual(left["coverage_hash"], right["coverage_hash"])
        self.assertEqual(left["sources"], right["sources"])

    def test_coverage_matrix_flags_mutation_without_resource(self):
        event = next(item for item in self.events if item.is_mutation)
        altered = event.__class__(**{**event.__dict__, "resource": None})
        result = self.diagnostics.coverage_matrix([altered])
        self.assertEqual(result["severity"], "critical")
        self.assertEqual(result["blind_spots"]["mutations_without_resource"], [event.event_id])

    def test_coverage_matrix_flags_terminal_without_correlation(self):
        event = next(item for item in self.events if item.is_terminal)
        altered = event.__class__(**{**event.__dict__, "correlation_id": None})
        result = self.diagnostics.coverage_matrix([altered])
        self.assertIn(event.event_id, result["blind_spots"]["terminal_without_correlation"])
        self.assertIn(result["severity"], {"warning", "critical"})

    def test_coverage_matrix_tracks_single_source_permissions(self):
        result = self.diagnostics.coverage_matrix(self.events[:1])
        for permission in self.events[0].permissions:
            self.assertEqual(result["blind_spots"]["single_source_permissions"][permission],
                             [self.events[0].source_id])

    def test_coverage_matrix_empty_input_is_clear(self):
        result = self.diagnostics.coverage_matrix([])
        self.assertEqual(result["severity"], "clear")
        self.assertEqual(result["event_count"], 0)
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
