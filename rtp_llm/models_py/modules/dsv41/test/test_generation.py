"""Real local attention/state checks; full-weight generation has a separate target."""

import unittest
from unittest import mock

import test_prefill as fixture
import torch
from rtp_llm.models.multimodal.deepseek_v41_processor import V41PreparedInputs
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.generation import V41TargetContinuation
from rtp_llm.models_py.modules.dsv41.inputs import V41CanonicalInputs
from rtp_llm.models_py.modules.dsv41.prefill import V41PrefillExecutor


class TargetContinuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.PrefillGpuTest.setUpClass()
        cls.helper = fixture.PrefillGpuTest()

    @classmethod
    def tearDownClass(cls):
        fixture.PrefillGpuTest.tearDownClass()

    def prefill(self, mode=ReplayMode.BOUNDED):
        tokens = (7, 9, 3, 6, 2)
        prepared = V41PreparedInputs("", tokens, (-1,) * len(tokens), ())
        cache = self.helper.cache(16, mode=mode)
        executor = V41PrefillExecutor.from_prepared(
            self.helper.target,
            cache,
            prepared,
            config=ReplayConfig(mode),
            chunk_tokens=128,
            draft_commit=self.helper.draft,
        )
        result = executor.run_next(epoch=0)
        return executor, result

    def test_full_and_bounded_preserve_canonical_history_and_full_decode(self):
        for mode in (ReplayMode.FULL, ReplayMode.BOUNDED):
            executor, result = self.prefill(mode)
            session = V41TargetContinuation.from_prefill(executor, result)
            tokens = executor.canonical.token_ids.tolist()
            for token in (11, 13, 17, 19):
                before = session.materialized_end
                tokens.append(token)
                expected = V41CanonicalInputs(
                    V41PreparedInputs("", tuple(tokens), (-1,) * len(tokens), ())
                ).rows(before, before + 1, device="cuda")
                with mock.patch.object(
                    session.target, "forward", wraps=session.target.forward
                ) as forward:
                    _, context = session.advance(token)
                    actual = forward.call_args.args[0]
                    torch.testing.assert_close(actual.history_ids, expected.history_ids)
                    torch.testing.assert_close(
                        actual.history_valid, expected.history_valid
                    )
                self.assertEqual(context.completed_layers, set(range(40)))
                self.assertEqual(session.materialized_end, before + 1)
                self.assertEqual(session.history_ids.numel(), 3)
                self.assertEqual(session.identity, executor.plan.identity)

    def test_idle_step_does_not_materialize_or_forget_current_output(self):
        executor, result = self.prefill()
        session = V41TargetContinuation.from_prefill(executor, result)
        before = session.history_ids.clone()
        output, context = session.advance(None)
        self.assertEqual(output.hidden_states.shape[0], 0)
        self.assertEqual(context.completed_layers, set(range(40)))
        self.assertEqual(session.materialized_end, 5)
        self.assertIs(session.output, result.output)
        torch.testing.assert_close(session.history_ids, before)

    def test_bad_token_and_capacity_are_rejected_before_mutating_cache(self):
        executor, result = self.prefill()
        session = V41TargetContinuation.from_prefill(executor, result)
        for token in (True, -1, 129280, 3.0):
            with self.assertRaises(ValueError):
                session.advance(token)
        self.assertEqual(session.cache.active_epoch, 0)
        session.cache.max_tokens = 5
        with self.assertRaises(ValueError):
            session.advance(3)
        self.assertFalse(session.cache.poisoned)

    def test_failed_or_incomplete_decode_poisons_cache_without_committing_history(self):
        for kind in ("exception", "incomplete"):
            executor, result = self.prefill()
            session = V41TargetContinuation.from_prefill(executor, result)
            before = session.history_ids.clone()
            options = (
                {"side_effect": RuntimeError("injected target failure")}
                if kind == "exception"
                else {"return_value": result.output}
            )
            with mock.patch.object(session.target, "forward", **options):
                with self.assertRaises(RuntimeError):
                    session.advance(11)
            self.assertTrue(session.cache.poisoned)
            self.assertEqual(session.materialized_end, 5)
            torch.testing.assert_close(session.history_ids, before)
            with self.assertRaises(RuntimeError):
                session.advance(13)

    def test_external_cache_use_and_incomplete_handoff_are_rejected(self):
        executor, result = self.prefill()
        session = V41TargetContinuation.from_prefill(executor, result)
        executor.cache.begin_forward(epoch=1, start=5, end=5)
        with self.assertRaises(RuntimeError):
            session.advance(11)
        with self.assertRaises(ValueError):
            V41TargetContinuation.from_prefill(executor, result)
        executor, result = self.prefill()
        executor.cache.swa_ends.pop(42)
        with self.assertRaises(ValueError):
            V41TargetContinuation.from_prefill(executor, result)


if __name__ == "__main__":
    unittest.main()
