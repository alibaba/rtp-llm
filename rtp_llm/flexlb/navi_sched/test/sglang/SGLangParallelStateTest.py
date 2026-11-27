import unittest

from navi_sched.adapters.sglang.SGLangParallelState import SGLangParallelState


def policy(**kwargs):
    return dict(
        total_blocks=100,
        page_size=64,
        linear_step=1,
        hybrid_tail_required=True,
        latency_fingerprint="same",
        **kwargs
    )


class SGLangParallelStateTest(unittest.TestCase):
    def state(self):
        state = SGLangParallelState(2)
        self.assertIsNone(state.ready(0, policy()))
        self.assertEqual(state.ready(1, policy())["total_blocks"], 200)
        return state

    def test_tail_on_other_rank_does_not_make_full_page_reusable(self):
        state = self.state()
        state.cache_delta(0, [[7], [], [7], [], [], []])
        delta = state.cache_delta(1, [[], [], [], [], [7], []])
        self.assertEqual(delta, [[], [], [], [], [], []])
        self.assertEqual(state.reserve_rank("request", [7], 100), 0)

    def test_eviction_retains_another_ranks_cached_page(self):
        state = self.state()
        for rank in range(2):
            state.cache_delta(rank, [[7], [], [7], [], [7], []])
        self.assertEqual(
            state.cache_delta(0, [[], [7], [], [7], [], [7]]), [[], [], [], [], [], []]
        )
        self.assertEqual(
            state.cache_delta(1, [[], [7], [], [7], [], [7]]),
            [[], [7], [], [7], [], [7]],
        )

    def test_affinity_selects_rank_with_full_prefix_and_its_own_tail(self):
        state = self.state()
        state.cache_delta(0, [[1], [], [1], [], [1], []])
        state.cache_delta(1, [[1, 2], [], [1, 2], [], [2], []])
        self.assertEqual(state.reserve_rank("warm", [1, 2], 200), 1)
        state.release_rank("warm")
        self.assertFalse(state.pending)

    def test_cold_requests_balance_before_scheduler_observes_them(self):
        state = self.state()
        self.assertEqual(state.reserve_rank("a", [], 20), 0)
        self.assertEqual(state.reserve_rank("b", [], 20), 1)
        state.release_rank("a")
        self.assertEqual(state.reserve_rank("c", [], 20), 0)

    def test_workload_replaces_previous_rank_sample_and_uses_latest_time(self):
        state = self.state()
        state.workload(0, {"event_time_us": 20, "request_count": 2}, ["a", "b"])
        value, ids = state.workload(1, {"event_time_us": 25, "request_count": 1}, ["c"])
        self.assertEqual(value, {"event_time_us": 25, "request_count": 3})
        self.assertEqual(ids, ["a", "b", "c"])
        value, ids = state.workload(0, {"event_time_us": 30, "request_count": 0}, [])
        self.assertEqual(value["request_count"], 1)
        self.assertEqual(ids, ["c"])

    def test_inconsistent_policy_prevents_readiness(self):
        state = SGLangParallelState(2)
        state.ready(0, policy())
        changed = policy()
        changed["page_size"] = 128
        with self.assertRaisesRegex(ValueError, "page_size"):
            state.ready(1, changed)
        with self.assertRaises(RuntimeError):
            state.reserve_rank("request", [], 1)

    def test_invalid_delta_keeps_previous_cache_state(self):
        state = self.state()
        with self.assertRaisesRegex(ValueError, "add and remove"):
            state.cache_delta(0, [[1], [], [2], [2], [], []])
        self.assertEqual(state.caches[0], [set(), set(), set()])

    def test_ready_preserves_cache_observed_before_last_rank_started(self):
        state = SGLangParallelState(2)
        first = dict(policy(), kind="ready", dp_rank=0)
        self.assertEqual(state.reduce(first), [])
        self.assertEqual(
            state.reduce(
                {
                    "kind": "cache_ready",
                    "dp_rank": 0,
                    "values": [[8], [], [8], [], [8], []],
                }
            ),
            [],
        )
        messages = state.reduce(dict(policy(), kind="ready", dp_rank=1))
        self.assertEqual(messages[0]["total_blocks"], 200)
        self.assertEqual(
            messages[1], {"kind": "cache_ready", "values": [[8], [], [8], [], [8], []]}
        )

    def test_overlapping_dp_execution_has_one_node_busy_window(self):
        state = self.state()
        state.policy = policy()
        start = {"kind": "execution_started", "request_count": 1}
        end = {"kind": "execution_finished"}
        self.assertEqual(len(state.reduce(dict(start, dp_rank=0))), 1)
        self.assertEqual(state.reduce(dict(start, dp_rank=1)), [])
        self.assertEqual(state.reduce(dict(end, dp_rank=0)), [])
        self.assertEqual(state.reduce(dict(end, dp_rank=1)), [end])
        self.assertEqual(state.execution_depth, {0: 0, 1: 0})


if __name__ == "__main__":
    unittest.main()
