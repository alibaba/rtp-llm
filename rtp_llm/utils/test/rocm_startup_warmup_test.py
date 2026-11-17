import unittest

from rtp_llm.utils.rocm_startup_warmup import rocm_warmup_cases, run_rocm_warmup_cases


class RocmWarmupPlanTest(unittest.TestCase):
    def test_bounded_plan_covers_cache_paths_and_decode(self):
        cases = rocm_warmup_cases(8192)
        self.assertEqual(len(cases), 8)
        self.assertFalse(cases[0].reuse_cache)
        self.assertTrue(cases[1].reuse_cache)
        self.assertEqual(
            [len(c.token_ids) - c.expected_reuse for c in cases[2:]],
            [1, 2, 3, 4, 8, 16],
        )
        for case in cases:
            self.assertLessEqual(len(case.token_ids) + case.max_new_tokens, 8192)
            self.assertEqual(case.max_new_tokens, 2)
        self.assertEqual(len({c.token_ids[256] for c in cases[2:]}), 6)
        self.assertTrue(all(c.token_ids[:256] == cases[1].token_ids for c in cases[2:]))

    def test_short_context_is_rejected_instead_of_skipping_coverage(self):
        with self.assertRaises(ValueError):
            rocm_warmup_cases(273)
        self.assertEqual(len(rocm_warmup_cases(274)), 8)


class RocmWarmupExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_cases_must_complete_in_order(self):
        seen = []
        cases = rocm_warmup_cases(8192)

        async def run(case):
            seen.append(case.name)
            return 2, case.expected_reuse

        await run_rocm_warmup_cases(cases, run)
        self.assertEqual(seen, [c.name for c in cases])

    async def test_early_eos_is_not_successful_decode_warmup(self):
        async def run(case):
            return 1, case.expected_reuse

        with self.assertRaisesRegex(RuntimeError, "did not exercise decode"):
            await run_rocm_warmup_cases(rocm_warmup_cases(8192), run)

    async def test_cache_miss_fails_closed(self):
        async def run(case):
            return 2, 0

        with self.assertRaisesRegex(RuntimeError, "cache path mismatch"):
            await run_rocm_warmup_cases(rocm_warmup_cases(8192), run)

    async def test_rpc_failure_is_not_swallowed(self):
        async def run(case):
            raise TimeoutError("compile timeout")

        with self.assertRaises(TimeoutError):
            await run_rocm_warmup_cases(rocm_warmup_cases(8192), run)


if __name__ == "__main__":
    unittest.main()
