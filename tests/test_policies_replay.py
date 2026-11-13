from __future__ import annotations

import unittest

from replay_lab.errors import PolicyError, ReplayError
from replay_lab.models import ReplayState
from replay_lab.policies import PolicyRuntime, PolicySandbox, parse_policy_sets
from replay_lab.replay import ReplayEngine

from tests.support import clean_events, event, incident_events, policy, timestamp


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.state = ReplayState("inc-001")

    def decide(self, spec, value):
        sandbox = PolicySandbox([spec])
        return sandbox.decide(value, self.state, sandbox.new_runtime())

    def test_permission_contraction_blocks_forbidden(self):
        spec = policy("permission_contraction", forbidden_permissions=["payments.send"], mode="block")
        value = event("e", 1, "tool_call_started", 1, permissions=["payments.send"])
        decision = self.decide(spec, value)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.action, "permission_block")

    def test_permission_contraction_shrinks_mixed_permissions(self):
        spec = policy("permission_contraction", forbidden_permissions=["payments.send"], mode="shrink")
        value = event("e", 1, "tool_call_started", 1, permissions=["payments.send", "calendar.read"])
        decision = self.decide(spec, value)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.mutations["effective_permissions"], ["calendar.read"])

    def test_permission_contraction_can_scope_agents(self):
        spec = policy("permission_contraction", forbidden_permissions=["payments.send"], agents=["other"])
        value = event("e", 1, "tool_call_started", 1, permissions=["payments.send"])
        self.assertTrue(self.decide(spec, value).allowed)

    def test_call_block_matches_tool(self):
        spec = policy("call_block", tools=["transfer"])
        value = event("e", 1, "tool_call_requested", 1, tool="transfer")
        decision = self.decide(spec, value)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.action, "call_block")

    def test_call_block_matches_permission(self):
        spec = policy("call_block", permissions=["payments.send"])
        value = event("e", 1, "tool_call_started", 1, permissions=["payments.send"])
        self.assertFalse(self.decide(spec, value).allowed)

    def test_call_block_no_filters_blocks_calls_only(self):
        spec = policy("call_block")
        call = event("e", 1, "tool_call_started", 1)
        resource = event("r", 2, "resource_read", 2)
        self.assertFalse(self.decide(spec, call).allowed)
        self.assertTrue(self.decide(spec, resource).allowed)

    def test_approval_gate_blocks_missing_correlation(self):
        spec = policy("approval_gate", permissions=["payments.send"])
        value = event("e", 1, "tool_call_started", 1, permissions=["payments.send"])
        self.assertFalse(self.decide(spec, value).allowed)

    def test_approval_gate_blocks_without_approval(self):
        sandbox = PolicySandbox([policy("approval_gate", permissions=["payments.send"])])
        runtime = sandbox.new_runtime()
        value = event("e", 1, "tool_call_started", 2, correlation_id="c", permissions=["payments.send"])
        self.assertFalse(sandbox.decide(value, self.state, runtime).allowed)

    def test_approval_gate_allows_prior_approval(self):
        sandbox = PolicySandbox([policy("approval_gate", permissions=["payments.send"], review_delay_ms=200)])
        runtime = sandbox.new_runtime()
        approval = event("a", 1, "approval_granted", 1, correlation_id="c", permissions=["payments.send"])
        sandbox.observe(approval, sandbox.decide(approval, self.state, runtime), runtime)
        value = event("e", 2, "tool_call_started", 2, correlation_id="c", permissions=["payments.send"])
        decision = sandbox.decide(value, self.state, runtime)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.delay_ms, 200)

    def test_approval_gate_rejects_expired_approval(self):
        sandbox = PolicySandbox([policy("approval_gate", permissions=["payments.send"], max_age_ms=1000)])
        runtime = sandbox.new_runtime()
        approval = event("a", 1, "approval_granted", 1, correlation_id="c", permissions=["payments.send"])
        sandbox.observe(approval, sandbox.decide(approval, self.state, runtime), runtime)
        value = event("e", 2, "tool_call_started", 5, correlation_id="c", permissions=["payments.send"])
        self.assertFalse(sandbox.decide(value, self.state, runtime).allowed)

    def test_rate_limit_blocks_second_call(self):
        sandbox = PolicySandbox([policy("rate_limit", limit=1, window_ms=10000, scope="agent", mode="block")])
        runtime = sandbox.new_runtime()
        first = event("a", 1, "tool_call_started", 1, tool="x")
        decision = sandbox.decide(first, self.state, runtime)
        sandbox.observe(first, decision, runtime)
        second = event("b", 2, "tool_call_started", 2, tool="y")
        self.assertFalse(sandbox.decide(second, self.state, runtime).allowed)

    def test_rate_limit_delay_returns_wait(self):
        sandbox = PolicySandbox([policy("rate_limit", limit=1, window_ms=10000, scope="agent", mode="delay")])
        runtime = sandbox.new_runtime()
        first = event("a", 1, "tool_call_started", 1, tool="x")
        sandbox.observe(first, sandbox.decide(first, self.state, runtime), runtime)
        second = event("b", 2, "tool_call_started", 2, tool="y")
        decision = sandbox.decide(second, self.state, runtime)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.action, "rate_delay")
        self.assertGreater(decision.delay_ms, 0)

    def test_rate_limit_expires_old_entries(self):
        sandbox = PolicySandbox([policy("rate_limit", limit=1, window_ms=1000, scope="agent")])
        runtime = sandbox.new_runtime()
        first = event("a", 1, "tool_call_started", 1)
        sandbox.observe(first, sandbox.decide(first, self.state, runtime), runtime)
        second = event("b", 2, "tool_call_started", 3)
        self.assertTrue(sandbox.decide(second, self.state, runtime).allowed)

    def test_rate_limit_invalid_values(self):
        sandbox = PolicySandbox([policy("rate_limit", limit=0, window_ms=1)])
        with self.assertRaises(PolicyError):
            sandbox.decide(event("a", 1, "tool_call_started", 1), self.state, sandbox.new_runtime())

    def test_application_isolation_blocks_application(self):
        spec = policy("application_isolation", applications=["wallet"], allow_event_kinds=[])
        value = event("e", 1, "tool_call_started", 1, application="wallet")
        self.assertFalse(self.decide(spec, value).allowed)

    def test_application_isolation_allows_listed_kind(self):
        spec = policy("application_isolation", applications=["wallet"], allow_event_kinds=["resource_read"])
        value = event("e", 1, "resource_read", 1, application="wallet")
        self.assertTrue(self.decide(spec, value).allowed)

    def test_policy_order_stops_at_first_block(self):
        sandbox = PolicySandbox([
            policy("permission_contraction", policy_id="a", priority=10, forbidden_permissions=["payments.send"]),
            policy("call_block", policy_id="b", priority=20, tools=["transfer"]),
        ])
        value = event("e", 1, "tool_call_started", 1, permissions=["payments.send"], tool="transfer")
        decision = sandbox.decide(value, self.state, sandbox.new_runtime())
        self.assertEqual(decision.policy_ids, ("a",))

    def test_unknown_policy_rejected(self):
        with self.assertRaises(PolicyError):
            PolicySandbox([policy("unknown")])

    def test_parse_policy_sets(self):
        result = parse_policy_sets([{"scenario_id": "x", "policies": [
            {"policy_id": "p", "kind": "call_block", "priority": 1, "enabled": True, "parameters": {}}
        ]}])
        self.assertEqual(result["x"][0].policy_id, "p")

    def test_parse_policy_sets_rejects_duplicates(self):
        with self.assertRaises(PolicyError):
            parse_policy_sets([{"scenario_id": "x", "policies": []}, {"scenario_id": "x", "policies": []}])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.engine = ReplayEngine()
        self.events = clean_events()

    def test_baseline_applies_all_events(self):
        result = self.engine.replay(self.events, scenario_id="baseline")
        self.assertEqual(result.metrics.applied_events, len(self.events))
        self.assertEqual(result.metrics.blocked_events, 0)
        self.assertIn("wallet:ledger:entry-77", result.state.resources)

    def test_baseline_loss_is_sum_of_events(self):
        result = self.engine.replay(self.events, scenario_id="baseline")
        self.assertEqual(result.metrics.loss_proxy, sum(item.loss_proxy for item in self.events))

    def test_permission_policy_reduces_loss(self):
        baseline = self.engine.replay(self.events, scenario_id="base")
        sandbox = PolicySandbox([policy("permission_contraction", forbidden_permissions=["payments.send"])])
        contained = self.engine.replay(self.events, scenario_id="contained", policies=sandbox)
        self.assertLess(contained.metrics.loss_proxy, baseline.metrics.loss_proxy)
        self.assertGreater(contained.metrics.blocked_events, 0)

    def test_call_block_prevents_resource_mutation(self):
        sandbox = PolicySandbox([policy("call_block", tools=["transfer"])])
        result = self.engine.replay(self.events, scenario_id="blocked", policies=sandbox)
        self.assertNotIn("wallet:ledger:entry-77", result.state.resources)

    def test_isolation_increases_availability_cost(self):
        sandbox = PolicySandbox([policy("application_isolation", applications=["wallet"], allow_event_kinds=[])])
        result = self.engine.replay(self.events, scenario_id="isolated", policies=sandbox)
        self.assertGreater(result.metrics.availability_cost, 0)

    def test_rate_delay_accumulates_response_delay(self):
        sandbox = PolicySandbox([policy("rate_limit", limit=1, window_ms=10000, scope="agent", mode="delay")])
        result = self.engine.replay(self.events, scenario_id="rate", policies=sandbox)
        self.assertGreater(result.metrics.response_delay_ms, 0)

    def test_deterministic_replay_hashes(self):
        left = self.engine.replay(self.events, scenario_id="same")
        right = self.engine.replay(reversed(self.events), scenario_id="same")
        self.assertEqual(left.state_hash, right.state_hash)
        self.assertEqual(left.run_id, right.run_id)

    def test_stop_after_limits_stream(self):
        result = self.engine.replay(self.events, scenario_id="partial", stop_after="e08")
        self.assertIn("e08", result.processed_event_ids)
        self.assertNotIn("e09", result.processed_event_ids)

    def test_synthetic_can_be_excluded(self):
        synthetic = event("hyp", 100, "resource_written", 10, resource="x", synthetic=True, confidence=0.4)
        result = self.engine.replay(self.events + [synthetic], scenario_id="real", include_synthetic=False)
        self.assertNotIn("x", result.state.resources)
        self.assertIn("hyp", result.skipped_event_ids)

    def test_empty_stream_rejected(self):
        with self.assertRaises(ReplayError):
            self.engine.replay([], scenario_id="empty")

    def test_multiple_incidents_rejected(self):
        other = event("other", 20, "agent_started", 20, incident_id="inc-002")
        with self.assertRaises(ReplayError):
            self.engine.replay(self.events + [other], scenario_id="bad")

    def test_checkpoint_round_trip(self):
        prefix = self.events[:8]
        prefix_result = self.engine.replay(prefix, scenario_id="prefix")
        checkpoint = self.engine.checkpoint(prefix_result, "cp", prefix[-1].event_time)
        resumed = self.engine.replay(self.events, scenario_id="resume", checkpoint=checkpoint)
        direct = self.engine.replay(self.events, scenario_id="resume")
        self.assertEqual(resumed.state.resources, direct.state.resources)
        self.assertEqual(resumed.state.tasks, direct.state.tasks)

    def test_checkpoint_hash_mismatch_rejected(self):
        prefix_result = self.engine.replay(self.events[:2], scenario_id="prefix")
        checkpoint = self.engine.checkpoint(prefix_result, "cp", self.events[1].event_time)
        bad = checkpoint.__class__(checkpoint.checkpoint_id, checkpoint.incident_id, checkpoint.event_time,
                                   checkpoint.last_event_ids, checkpoint.state, "0" * 64, checkpoint.source)
        with self.assertRaises(Exception):
            self.engine.replay(self.events, scenario_id="bad", checkpoint=bad)

    def test_timeline_has_one_entry_per_decision(self):
        result = self.engine.replay(self.events, scenario_id="timeline")
        self.assertEqual(len(result.state.timeline), len(result.decisions))
        self.assertEqual(result.state.timeline[-1]["event_id"], "e14")

    def test_state_counters_track_kinds(self):
        result = self.engine.replay(self.events, scenario_id="counters")
        self.assertEqual(result.state.counters["tool_call_started"], 2)
        self.assertEqual(result.state.counters["agent_started"], 1)

    def test_resource_record_tracks_agent_and_outcome(self):
        result = self.engine.replay(self.events, scenario_id="resources")
        record = result.state.resources["wallet:ledger:entry-77"]
        self.assertEqual(record["agents"], ["agent-alpha"])
        self.assertEqual(record["writes"], 1)
        self.assertEqual(record["last_outcome"], "ok")


if __name__ == "__main__":
    unittest.main()
