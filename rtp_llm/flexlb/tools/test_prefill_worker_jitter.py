"""Fast contract tests; no Java process or network connection is needed."""

import copy
import unittest

from prefill_worker_jitter import FACTORS, JitterController, audit_execution, factor


class FakeOps:
    mock_http_port = 12345

    def __init__(self):
        self.engines = [
            dict(name="prefill-0", engine_incarnation="generation-0", role="prefill"),
            dict(name="decode-0", engine_incarnation="generation-d", role="decode"),
        ]

    def snapshot(self):
        return {"engines": self.engines}


class FakeHttp:
    def __init__(self):
        self.states = {}
        self.posts = []

    def post(self, url, body):
        self.posts.append(body)
        state = dict(expression=body["expression"], runtime_override=True)
        self.states[body["engine"]] = state
        return 200, {"engines": {body["engine"]: state}}

    def get(self, url):
        return {"engines": self.states}


def evidence(mode="independent", members=1):
    records, events = [], []
    for slot in range(10):
        multiplier = factor(42, "prefill-0", slot, mode)
        expression = f"sum(computeTokens) / 70 / {multiplier}"
        state = dict(expression=expression, runtime_override=True)
        records.append(
            dict(
                engine="prefill-0",
                engine_incarnation="generation-0",
                slot=slot,
                mode=mode,
                factor=multiplier,
                expression=expression,
                status=200,
                request_wall_ms=slot * 1000,
                ack_wall_ms=slot * 1000 + 5,
                readback=state,
            )
        )
        # 7k computed tokens per batch; cache hits must not enter execution cost.
        duration = max(1, int(100 / multiplier))
        for member in range(members):
            events.append(
                dict(
                    event="prefill_done",
                    engine_name="prefill-0",
                    engine_incarnation="generation-0",
                    rid=slot * 10 + member,
                    prefill_start_ms=slot * 1000 + 100 + member,
                    prefill_done_ms=slot * 1000 + 100 + duration + 2,
                    exec_ms=duration,
                    batch_size=members,
                    input_len=14_000 // members,
                    cache_hit_tokens=7_000 // members,
                    cancelled=False,
                    batch_id=slot * 100 + member,
                )
            )
    return events, records


class FactorTest(unittest.TestCase):
    def test_each_five_slots_exercises_all_factors(self):
        for seed in (0, 1, 42, -10):
            for worker in ("prefill-0", "prefill-1", "新增-worker"):
                for cycle in range(10):
                    self.assertEqual(
                        set(FACTORS),
                        {
                            factor(seed, worker, cycle * 5 + offset, "independent")
                            for offset in range(5)
                        },
                    )

    def test_independent_workers_have_distinct_but_repeatable_sequences(self):
        def sequence(worker):
            return [factor(42, worker, slot, "independent") for slot in range(50)]

        self.assertEqual(sequence("prefill-0"), sequence("prefill-0"))
        self.assertNotEqual(sequence("prefill-0"), sequence("prefill-1"))

    def test_correlated_workers_share_sequence(self):
        for slot in range(50):
            self.assertEqual(
                factor(3, "prefill-0", slot, "correlated"),
                factor(3, "prefill-91", slot, "correlated"),
            )

    def test_stable_and_invalid_input(self):
        self.assertEqual(1.0, factor(2, "worker", 30, "stable"))
        with self.assertRaises(ValueError):
            factor(1, "worker", -1, "independent")
        with self.assertRaises(ValueError):
            factor(1, "worker", 0, "typo")


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.ops, self.http = FakeOps(), FakeHttp()
        self.controller = JitterController(
            self.ops, http_get=self.http.get, http_post=self.http.post
        )

    def test_per_worker_formula_and_readback(self):
        self.controller.apply_slot(0)
        self.assertEqual(1, len(self.http.posts))
        self.assertEqual("prefill-0", self.http.posts[0]["engine"])
        row = self.controller.records[0]
        self.assertEqual(row["expression"], row["readback"]["expression"])
        self.assertLessEqual(row["request_mono_s"], row["ack_mono_s"])
        self.assertLessEqual(row["ack_wall_ms"], row["readback_wall_ms"])

    def test_new_and_restarted_generations_get_controls(self):
        self.controller.apply_slot(0)
        self.ops.engines.append(
            dict(name="prefill-1", engine_incarnation="generation-1", role="prefill")
        )
        self.ops.engines[0]["engine_incarnation"] = "replacement-0"
        self.controller.apply_slot(1)
        self.assertEqual(
            {("prefill-0", "replacement-0"), ("prefill-1", "generation-1")},
            {
                (row["engine"], row["engine_incarnation"])
                for row in self.controller.records[1:]
            },
        )

    def test_http_failure_and_wrong_ack_fail(self):
        self.controller.http_post = lambda *args: (500, {"error": "failure"})
        with self.assertRaisesRegex(RuntimeError, "update failed"):
            self.controller.apply_slot(0)
        self.controller.http_post = lambda *args: (200, {"engines": {}})
        with self.assertRaisesRegex(RuntimeError, "acknowledgement disagrees"):
            self.controller.apply_slot(0)

    def test_readback_disagreement_is_not_hidden_by_http_success(self):
        self.controller.http_get = lambda *args: {"engines": {}}
        with self.assertRaisesRegex(RuntimeError, "readback disagrees"):
            self.controller.apply_slot(0)

    def test_graceful_removal_between_snapshot_and_control_is_recorded(self):
        def remove(*args):
            self.ops.engines = []
            return 400, {"error": "no matching prefill engine"}

        self.controller.http_post = remove
        self.controller.apply_slot(0)
        self.assertTrue(self.controller.records[0]["removed_before_apply"])

    def test_background_errors_surface_at_stop(self):
        self.controller.errors.append("deliberate failure")
        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            self.controller.stop()

    def test_start_applies_initial_slot_and_stop_joins_thread(self):
        self.assertIs(self.controller, self.controller.start())
        records = self.controller.stop()
        self.assertEqual(1, len(records))
        self.assertFalse(self.controller._thread.is_alive())
        self.assertEqual(0, records[0]["slot"])
        with self.assertRaisesRegex(RuntimeError, "only be started once"):
            self.controller.start()

    def test_invalid_period_and_mode(self):
        for period in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                JitterController(self.ops, period_s=period)
        with self.assertRaises(ValueError):
            JitterController(self.ops, mode="invalid")


class ExecutionAuditTest(unittest.TestCase):
    def test_nonbatch_and_regrouped_batch_use_same_compute_cost(self):
        for mode in ("stable", "independent", "correlated"):
            single, records = evidence(mode)
            batched, _ = evidence(mode, members=2)
            first = audit_execution(single, records, mode)
            second = audit_execution(batched, records, mode)
            self.assertEqual(10, second["checked_batches"])
            self.assertEqual(70_000, second["executed_compute_tokens"])
            self.assertEqual(first["modeled_busy_ms"], second["modeled_busy_ms"])

    def test_cancelled_members_remain_part_of_actual_batch(self):
        events, records = evidence(members=2)
        events[0]["cancelled"] = True
        self.assertEqual(
            70_000,
            audit_execution(events, records, "independent")["executed_compute_tokens"],
        )
        with self.assertRaisesRegex(AssertionError, "incomplete"):
            audit_execution(events[1:], records, "independent")

    def test_fake_jitter_that_only_updates_controls_fails(self):
        events, records = evidence()
        for event in events:
            event["exec_ms"] = 100
            event["prefill_done_ms"] = event["prefill_start_ms"] + 102
        with self.assertRaisesRegex(AssertionError, "disagrees"):
            audit_execution(events, records, "independent")

    def test_control_evidence_must_match_declared_factor_and_mode(self):
        events, records = evidence()
        for field, value in (
            ("mode", "stable"),
            ("factor", 0.8),
            ("expression", "sum(computeTokens) / 70"),
            ("readback", {"expression": "wrong", "runtime_override": True}),
        ):
            invalid = copy.deepcopy(records)
            invalid[0][field] = value
            with self.assertRaises(AssertionError):
                audit_execution(events, invalid, "independent")

    def test_invalid_tokens_duplicate_and_duration_fail(self):
        events, records = evidence()
        cases = []
        invalid_tokens = copy.deepcopy(events)
        invalid_tokens[0]["cache_hit_tokens"] = 100_000
        cases.append(invalid_tokens)
        cases.append(events + [events[0]])
        invalid_duration = copy.deepcopy(events)
        invalid_duration[0]["exec_ms"] = 900
        cases.append(invalid_duration)
        for invalid in cases:
            with self.assertRaises(AssertionError):
                audit_execution(invalid, records, "independent")

    def test_uncontrolled_generation_fails(self):
        events, records = evidence()
        events[0]["engine_incarnation"] = "unknown"
        with self.assertRaisesRegex(AssertionError, "never controlled"):
            audit_execution(events, records, "independent")

    def test_transition_is_labeled_and_cannot_hide_whole_run(self):
        events, records = evidence()
        events[0]["prefill_start_ms"] = 5
        report = audit_execution(events, records, "independent")
        self.assertEqual(1, report["ambiguous_batches"])
        for event, control in zip(events, records):
            event["prefill_start_ms"] = control["ack_wall_ms"]
        with self.assertRaisesRegex(AssertionError, "too few|too many"):
            audit_execution(events, records, "independent")

    def test_conversion_boundary_still_checks_actual_factor_range(self):
        events, records = evidence()
        events[0]["prefill_start_ms"] = 5
        events[0]["exec_ms"] = 500
        with self.assertRaisesRegex(AssertionError, "configured throughput factor"):
            audit_execution(events, records, "independent")

    def test_warmup_is_excluded_and_both_extremes_required(self):
        events, records = evidence()
        warmup = dict(events[0], rid=-1, prefill_start_ms=-100, prefill_done_ms=-1)
        self.assertEqual(
            10,
            audit_execution([warmup] + events, records, "independent")["batches"],
        )
        only_fast = [
            event for event, control in zip(events, records) if control["factor"] == 1.1
        ]
        with self.assertRaisesRegex(AssertionError, "both"):
            audit_execution(only_fast, records, "independent", min_checked_batches=1)


if __name__ == "__main__":
    unittest.main()
