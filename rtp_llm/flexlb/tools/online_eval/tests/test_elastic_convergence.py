import unittest

from flexlb_test_framework.scenario.actions.elastic_convergence import (
    convergence_bound,
    dead_interval,
    evaluate_convergence,
)


class ConvergenceTest(unittest.TestCase):
    def event(self, operation, at, address="127.1.0.1:5000"):
        return dict(operation=operation, ended_s=at, address=address, ok=True)

    def request(self, at, address="127.1.0.2:5001", status="OK", elapsed=0.1):
        return dict(
            wire_request_id=at,
            issued_s=at,
            prefill_addr=address,
            schedule=dict(started_s=at, ended_s=at + elapsed, status=status),
            stream=dict(status="UNAVAILABLE"),
            transport_terminal_s=at + elapsed,
        )

    def evaluate(self, rows, ops, batch=False):
        return evaluate_convergence(
            rows,
            ops,
            10,
            15,
            batch=batch,
            min_post_samples=2,
            nonbatch_failfast_s=2,
            batch_failfast_s=5,
        )

    def test_normal_convergence_and_no_dead_hit_is_skip_evidence(self):
        result = self.evaluate(
            [self.request(26), self.request(27)], [self.event("remove", 9)]
        )
        self.assertTrue(result["coverage_ok"])
        self.assertEqual([], result["late_dead_hits"])
        self.assertEqual(0, result["failfast_samples"])

    def test_dead_hit_after_bound_fails(self):
        result = self.evaluate(
            [self.request(26, "127.1.0.1:5000")], [self.event("remove", 9)]
        )
        self.assertEqual(1, len(result["late_dead_hits"]))

    def test_exact_address_reuse_opens_new_generation(self):
        ops = [self.event("remove", 9), self.event("add", 20)]
        self.assertEqual(9, dead_interval(ops, "127.1.0.1:5000", 15))
        self.assertIsNone(dead_interval(ops, "127.1.0.1:5000", 21))
        self.assertIsNone(dead_interval(ops, "127.9.0.1:5000", 15))

    def test_batch_thirty_second_hang_without_address_fails(self):
        result = self.evaluate(
            [self.request(5, None, "DEADLINE_EXCEEDED", 30)],
            [self.event("remove", 4)],
            batch=True,
        )
        self.assertEqual([5], result["failfast_violations"])
        self.assertEqual([5], result["unattributed_batch_failures"])

    def test_no_post_bound_samples_never_vacuously_passes(self):
        self.assertFalse(self.evaluate([], [self.event("remove", 9)])["coverage_ok"])

    def test_configuration_drift_is_not_capped_away(self):
        config = {
            "workerRegistry": {
                "health": {"statusStaleAfterMs": 10000, "cleanupIntervalMs": 3000}
            }
        }
        self.assertEqual(15, convergence_bound(config, 2, 30))
        config["workerRegistry"]["health"]["statusStaleAfterMs"] = 40000
        with self.assertRaisesRegex(ValueError, "configuration drift"):
            convergence_bound(config, 2, 30)

    def test_late_remove_moves_quiescent_cutoff(self):
        result = self.evaluate(
            [self.request(26), self.request(27)], [self.event("remove", 20)]
        )
        self.assertEqual(35, result["cutoff_s"])
        self.assertFalse(result["coverage_ok"])

    def test_wildcard_listener_port_reuse_is_rejected_as_ambiguous(self):
        from types import SimpleNamespace

        from flexlb_test_framework.scenario.actions.elastic_concurrent import (
            crossfire_validate,
        )

        config = dict(
            margin_s=2,
            cap_s=30,
            post_window_s=10,
            min_post_samples=2,
            nonbatch_failfast_s=2,
            batch_failfast_s=5,
            unique_ports=False,
        )
        with self.assertRaisesRegex(ValueError, "unique ports"):
            crossfire_validate(dict(convergence=config), SimpleNamespace(path="test"))
        config["unique_ports"] = True
        self.assertTrue(
            crossfire_validate(dict(convergence=config), SimpleNamespace(path="test"))[
                "convergence"
            ]["unique_ports"]
        )
