from __future__ import annotations

import unittest

from replay_lab.causality import CausalityAnalyzer
from replay_lab.comparison import Comparator
from replay_lab.errors import ExperimentError
from replay_lab.experiments import ExperimentRunner, Scenario, scenarios_from_dict
from replay_lab.interpolation import Interpolator

from tests.support import incident_events, policy, policy_scenarios, timestamp


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.events = incident_events()
        report = CausalityAnalyzer(gap_threshold_ms=8000).analyze(self.events)
        self.candidates = Interpolator().generate(report.gaps, self.events)
        self.runner = ExperimentRunner()
        self.scenarios = (
            Scenario("baseline", "Baseline", ()),
            Scenario("blocked", "Blocked", (policy("call_block", tools=["transfer"]),)),
        )

    def plan(self, **overrides):
        values = dict(incident_id="inc-001", created_at=timestamp(40), seed=7, trials=4,
                      uncertainty_mode="sample_candidates", scenarios=self.scenarios, parameters={})
        values.update(overrides)
        return self.runner.build_plan(**values)

    def test_build_plan_generates_stable_id(self):
        self.assertEqual(self.plan().experiment_id, self.plan().experiment_id)

    def test_build_plan_rejects_unknown_mode(self):
        with self.assertRaises(ExperimentError):
            self.plan(uncertainty_mode="random")

    def test_build_plan_rejects_zero_trials(self):
        with self.assertRaises(ExperimentError):
            self.plan(trials=0)

    def test_build_plan_rejects_duplicate_scenarios(self):
        duplicate = (Scenario("x", "X", ()), Scenario("x", "Y", ()))
        with self.assertRaises(ExperimentError):
            self.plan(scenarios=duplicate)

    def test_sample_candidates_is_deterministic(self):
        plan = self.plan()
        left = self.runner.run(plan, self.events, self.candidates)
        right = self.runner.run(plan, self.events, self.candidates)
        self.assertEqual(left.experiment_hash, right.experiment_hash)
        self.assertEqual([trial.selected_candidate_ids for trial in left.trials],
                         [trial.selected_candidate_ids for trial in right.trials])

    def test_different_seed_can_change_selection(self):
        left = self.runner.run(self.plan(seed=1, trials=12), self.events, self.candidates)
        right = self.runner.run(self.plan(seed=2, trials=12), self.events, self.candidates)
        self.assertNotEqual([trial.selected_candidate_ids for trial in left.trials],
                            [trial.selected_candidate_ids for trial in right.trials])

    def test_evidence_only_selects_no_candidates(self):
        result = self.runner.run(self.plan(uncertainty_mode="evidence_only"), self.events, self.candidates)
        self.assertTrue(all(not trial.selected_candidate_ids for trial in result.trials))

    def test_best_candidate_selects_one_per_gap(self):
        result = self.runner.run(self.plan(uncertainty_mode="best_candidate"), self.events, self.candidates)
        expected = len({candidate.gap_id for candidate in self.candidates})
        self.assertTrue(all(len(trial.selected_candidate_ids) == expected for trial in result.trials))

    def test_all_candidates_selects_every_candidate(self):
        result = self.runner.run(self.plan(uncertainty_mode="all_candidates"), self.events, self.candidates)
        self.assertTrue(all(len(trial.selected_candidate_ids) == len(self.candidates) for trial in result.trials))

    def test_trial_count_multiplies_scenarios(self):
        result = self.runner.run(self.plan(trials=3), self.events, self.candidates)
        self.assertEqual(len(result.trials), 6)

    def test_summaries_contain_distributions(self):
        result = self.runner.run(self.plan(), self.events, self.candidates)
        summary = result.scenario_summaries["baseline"]
        self.assertEqual(summary["trials"], 4)
        self.assertIn("p90", summary["loss_proxy"])

    def test_blocked_scenario_reduces_loss(self):
        result = self.runner.run(self.plan(uncertainty_mode="evidence_only"), self.events, self.candidates)
        self.assertLess(result.scenario_summaries["blocked"]["loss_proxy"]["mean"],
                        result.scenario_summaries["baseline"]["loss_proxy"]["mean"])

    def test_scenarios_from_dict(self):
        values = scenarios_from_dict(policy_scenarios()[:2])
        self.assertEqual(values[0].scenario_id, "baseline")
        self.assertEqual(values[1].policies[0].kind, "permission_contraction")

    def test_scenarios_from_dict_requires_policies_array(self):
        with self.assertRaises(ExperimentError):
            scenarios_from_dict([{"scenario_id": "x", "policies": "bad"}])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        events = incident_events()
        report = CausalityAnalyzer(gap_threshold_ms=8000).analyze(events)
        candidates = Interpolator().generate(report.gaps, events)
        runner = ExperimentRunner()
        scenarios = (
            Scenario("baseline", "Baseline", ()),
            Scenario("permission", "Permission", (policy("permission_contraction", forbidden_permissions=["payments.send"]),)),
            Scenario("isolation", "Isolation", (policy("application_isolation", applications=["wallet"], allow_event_kinds=[]),)),
        )
        plan = runner.build_plan(incident_id="inc-001", created_at=timestamp(40), seed=4, trials=3,
                                 uncertainty_mode="evidence_only", scenarios=scenarios)
        self.result = runner.run(plan, events, candidates)
        self.comparator = Comparator()

    def test_compare_uses_requested_baseline(self):
        report = self.comparator.compare(self.result, baseline_id="baseline")
        self.assertEqual(report.baseline_id, "baseline")

    def test_compare_rejects_missing_baseline(self):
        with self.assertRaises(ValueError):
            self.comparator.compare(self.result, baseline_id="missing")

    def test_rankings_contain_all_scenarios(self):
        report = self.comparator.compare(self.result)
        self.assertEqual(set(report.rankings), set(self.result.scenario_summaries))

    def test_pareto_front_not_empty(self):
        report = self.comparator.compare(self.result)
        self.assertTrue(report.pareto_front)

    def test_comparison_hash_deterministic(self):
        left = self.comparator.compare(self.result)
        right = self.comparator.compare(self.result)
        self.assertEqual(left.comparison_hash, right.comparison_hash)

    def test_verify_recreates_default_report(self):
        report = self.comparator.compare(self.result)
        self.assertTrue(self.comparator.verify(report, self.result))

    def test_permission_has_positive_loss_reduction(self):
        report = self.comparator.compare(self.result)
        row = next(item for item in report.rows if item["scenario_id"] == "permission")
        self.assertGreater(row["loss_reduction"], 0)

    def test_rows_include_relative_change(self):
        report = self.comparator.compare(self.result)
        row = report.rows[0]
        self.assertIn("relative_change", row["loss_proxy"])


if __name__ == "__main__":
    unittest.main()
