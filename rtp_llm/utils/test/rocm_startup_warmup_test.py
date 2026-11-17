import asyncio
import unittest

from rtp_llm.utils.rocm_startup_warmup import (
    rocm_startup_warmup_enabled,
    rocm_warmup_cases,
    run_rocm_warmup_cases,
)


class RocmWarmupPolicyTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(
            is_rocm=True,
            model_type="qwen_3",
            is_pdfusion=True,
            world_size=1,
            reuse_cache=True,
            speculative=False,
        )

    def test_supported_service_is_enabled_without_environment_variable(self):
        for override in ("auto", "", "1", "true"):
            self.assertTrue(rocm_startup_warmup_enabled(override, **self.config))

    def test_explicit_opt_out(self):
        for override in ("0", "false", "off", "no"):
            self.assertFalse(rocm_startup_warmup_enabled(override, **self.config))

    def test_auto_preserves_other_platforms_and_topologies(self):
        for key, value in dict(
            is_rocm=False,
            model_type="qwen_3_moe",
            is_pdfusion=False,
            world_size=2,
            reuse_cache=False,
            speculative=True,
        ).items():
            with self.subTest(key=key):
                config = dict(self.config, **{key: value})
                self.assertFalse(rocm_startup_warmup_enabled("auto", **config))
                if key != "model_type":
                    with self.assertRaises(ValueError):
                        rocm_startup_warmup_enabled("1", **config)

    def test_invalid_override_fails_closed(self):
        with self.assertRaises(ValueError):
            rocm_startup_warmup_enabled("typo", **self.config)


class RocmWarmupPlanTest(unittest.TestCase):
    def test_bounded_plan_covers_cache_paths_and_decode(self):
        cases = rocm_warmup_cases(8192)
        self.assertEqual(len(cases), 10)
        self.assertFalse(cases[0].reuse_cache)
        self.assertTrue(cases[1].reuse_cache)
        self.assertEqual(
            [len(c.token_ids) - c.expected_reuse for c in cases[2:]],
            [1, 2, 3, 4, 8, 14, 15, 16],
        )
        for case in cases:
            self.assertLessEqual(len(case.token_ids) + case.max_new_tokens, 8192)
            self.assertEqual(case.max_new_tokens, 2)
        self.assertEqual(len({c.token_ids[256] for c in cases[2:]}), 8)
        self.assertTrue(all(c.token_ids[:256] == cases[1].token_ids for c in cases[2:]))

    def test_short_context_is_rejected_instead_of_skipping_coverage(self):
        with self.assertRaises(ValueError):
            rocm_warmup_cases(273)
        self.assertEqual(len(rocm_warmup_cases(274)), 10)


class RocmWarmupExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_hung_rpc_is_cancelled_and_no_next_case_runs(self):
        seen = []
        cancelled = []

        async def run(case):
            seen.append(case.name)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(case.name)

        with self.assertRaises(asyncio.TimeoutError):
            await run_rocm_warmup_cases(rocm_warmup_cases(8192), run, timeout_s=0.01)
        self.assertEqual(seen, ["uncached"])
        self.assertEqual(cancelled, seen)

    async def test_invalid_timeout_does_not_submit_requests(self):
        async def run(case):
            self.fail("invalid timeout must fail before sending a request")

        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                await run_rocm_warmup_cases(
                    rocm_warmup_cases(8192), run, timeout_s=timeout
                )

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
