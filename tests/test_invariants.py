from __future__ import annotations

import unittest

from replay_lab.causality import CausalityAnalyzer
from replay_lab.experiments import ExperimentRunner, Scenario
from replay_lab.interpolation import Interpolator
from replay_lab.invariants import InvariantChecker
from replay_lab.replay import ReplayEngine

from tests.support import clean_events, incident_events, policy, timestamp


class InvariantTests(unittest.TestCase):
    def test_valid_replay_passes(self):
        events = clean_events()
        result = ReplayEngine().replay(events, scenario_id="baseline")
        checked = InvariantChecker().check_replay(result, events)
        self.assertTrue(checked["ok"], checked)

    def test_valid_experiment_passes(self):
        events = incident_events()
        causal = CausalityAnalyzer(gap_threshold_ms=8000).analyze(events)
        candidates = Interpolator().generate(causal.gaps, events)
        scenarios = (Scenario("baseline", "Baseline", ()),
                     Scenario("block", "Block", (policy("call_block", tools=["transfer"]),)))
        runner = ExperimentRunner()
        plan = runner.build_plan(incident_id="inc-001", created_at=timestamp(40), seed=1,
                                 trials=2, uncertainty_mode="evidence_only", scenarios=scenarios)
        result = runner.run(plan, events, candidates)
        checked = InvariantChecker().check_experiment(result, events)
        self.assertTrue(checked["ok"], checked)


if __name__ == "__main__":
    unittest.main()
