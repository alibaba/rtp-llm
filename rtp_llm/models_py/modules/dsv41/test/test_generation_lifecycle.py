"""Local state failures for the target EP generation coordinator."""

import unittest
from unittest import mock

import test_generation as fixture
import torch
from fixture import flash_config
from rtp_llm.models_py.modules.dsv41.generation import (
    V41TargetContinuation,
    V41TargetGeneration,
)


class TargetGenerationLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.TargetContinuationTest.setUpClass()
        cls.helper = fixture.TargetContinuationTest()

    @classmethod
    def tearDownClass(cls):
        fixture.TargetContinuationTest.tearDownClass()

    def generation(self, budget=4):
        executor, result = self.helper.prefill()
        session = V41TargetContinuation.from_prefill(executor, result)
        session.target.config.eos_token_id = flash_config()["eos_token_id"]
        with mock.patch("torch.distributed.is_initialized", return_value=True):
            return V41TargetGeneration(session, max_output_tokens=budget)

    def logits(self, token):
        logits = torch.zeros((1, 129280), device="cuda")
        logits[0, token] = 1
        return logits

    def test_length_keeps_final_sample_unmaterialized_and_joins_empty_steps(self):
        generation = self.generation(1)
        session = generation.continuation
        with mock.patch.object(session, "logits", return_value=self.logits(11)):
            with mock.patch(
                "torch.distributed.all_reduce",
                side_effect=lambda active, **_: active.fill_(2),
            ):
                first = generation.step()
                second = generation.step()
        self.assertEqual((first.token_id, first.finish_reason), (11, "length"))
        self.assertIsNone(second.token_id)
        self.assertEqual(generation.generated_token_ids, [11])
        self.assertEqual(session.materialized_end, 5)
        for step in (first, second):
            self.assertEqual(step.context.completed_layers, set(range(40)))
            self.assertEqual(step.context.start, step.context.end)
        with mock.patch("torch.distributed.all_reduce"):
            last = generation.step()
        self.assertIsNone(last.context)
        self.assertTrue(generation.globally_complete)
        with self.assertRaises(RuntimeError):
            generation.step()

    def test_eos_and_cancel_never_resample_or_advance_the_local_boundary(self):
        for cancel in (False, True):
            generation = self.generation()
            session = generation.continuation
            if cancel:
                generation.cancel()
            with mock.patch.object(
                session,
                "logits",
                return_value=self.logits(session.target.config.eos_token_id),
            ) as logits:
                with mock.patch("torch.distributed.all_reduce"):
                    step = generation.step()
            self.assertEqual(step.finish_reason, "cancelled" if cancel else "stop")
            self.assertEqual(logits.call_count, 0 if cancel else 1)
            self.assertEqual(session.materialized_end, 5)
            self.assertTrue(generation.globally_complete)

    def test_bad_logits_or_collective_failure_poison_the_request(self):
        for error in ("logits", "collective"):
            generation = self.generation()
            session = generation.continuation
            logits = self.logits(11)
            if error == "logits":
                logits[0, 1] = float("nan")
            with mock.patch.object(session, "logits", return_value=logits):
                with mock.patch(
                    "torch.distributed.all_reduce",
                    side_effect=RuntimeError("transport failed"),
                ):
                    with self.assertRaises(RuntimeError):
                        generation.step()
            self.assertTrue(session.cache.poisoned)
            self.assertTrue(generation.failed)
            self.assertEqual(session.materialized_end, 5)
            with self.assertRaises(RuntimeError):
                generation.cancel()

    def test_budget_and_group_admission_precede_generation(self):
        session = self.generation().continuation
        with mock.patch("torch.distributed.is_initialized", return_value=True):
            for budget in (0, -1, True, 1.5, 12):
                with self.assertRaises(ValueError):
                    V41TargetGeneration(session, max_output_tokens=budget)
            with self.assertRaises(ValueError):
                V41TargetGeneration(
                    session, max_output_tokens=1, process_group=object()
                )
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            with self.assertRaises(RuntimeError):
                V41TargetGeneration(session, max_output_tokens=1)
        self.assertFalse(session.cache.poisoned)


if __name__ == "__main__":
    unittest.main()
