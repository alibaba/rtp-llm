from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import (
    _HC,
    _HIDDEN,
    PLERuntimeFixture,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.ple_page_reference import (
    _apply_ple as reference_apply_ple,
)


class Qwen4ExpPLEPrefetchTest(PLERuntimeFixture, TestCase):
    def test_async_prefetch_ragged_prefill_preserves_every_page_and_prefix(self):
        embedding = self._enable_async_cpu_ple()
        device = torch.device("cuda")
        with self.subTest(mode="automatic"):
            self.state_base.normal_()
            self.ctx_base.random_(1, 7)
            before_state, before_context = (
                self.state_base.clone(),
                self.ctx_base.clone(),
            )
            ids = torch.randint(0, 10, (26,), device=device, dtype=torch.long)
            hyper = torch.randn(26, _HC * _HIDDEN, device=device, dtype=torch.bfloat16)
            inputs = self._inputs_by_tag(
                is_prefill=True,
                input_lengths=torch.tensor([17, 9], dtype=torch.int32),
                prefix_lengths=torch.tensor([0, 8], dtype=torch.int32),
                sequence_lengths=torch.empty(0, dtype=torch.int32),
            )
            expected = reference_apply_ple(
                self.model._ple_execution(), 1, hyper, ids, inputs
            )
            expected_state, expected_context = (
                self.state_base.clone(),
                self.ctx_base.clone(),
            )
            self.state_base.copy_(before_state)
            self.ctx_base.copy_(before_context)
            plans = self.model._start_eager_ple_prefetch(
                SimpleNamespace(input_ids=ids), inputs
            )
            torch.testing.assert_close(self.state_base, before_state, atol=0, rtol=0)
            torch.testing.assert_close(self.ctx_base, before_context, atol=0, rtol=0)
            actual = self.model._apply_ple(1, hyper, ids, inputs, prefetch=plans[1])
            embedding.finish_prefetch()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(self.state_base, expected_state, atol=0, rtol=0)
            torch.testing.assert_close(self.ctx_base, expected_context, atol=0, rtol=0)

    def test_async_target_verify_commit_and_rollback_preserve_state(self):
        embedding = self._enable_async_cpu_ple()
        device = torch.device("cuda")
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        before_state, before_context = self.state_base.clone(), self.ctx_base.clone()
        ids = torch.tensor([[4, 5, 6], [3, 4, 5]], device=device, dtype=torch.long)
        hyper = torch.randn(6, _HC * _HIDDEN, device=device, dtype=torch.bfloat16)
        inputs = self._target_inputs(prefixes, 3)
        expected = self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 3], device=device, dtype=torch.int32)
        )
        self.model.finish_speculative_target_commit(True)
        committed_state, committed_context = (
            self.state_base.clone(),
            self.ctx_base.clone(),
        )
        self.model.finish_speculative_target_commit(False)
        plans = self.model._start_eager_ple_prefetch(
            SimpleNamespace(input_ids=ids.flatten()), inputs
        )
        actual = self.model._apply_ple(
            1, hyper, ids.flatten(), inputs, prefetch=plans[1]
        )
        embedding.finish_prefetch()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(self.state_base, before_state, atol=0, rtol=0)
        torch.testing.assert_close(self.ctx_base, before_context, atol=0, rtol=0)

        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 3], device=device, dtype=torch.int32)
        )
        self.model.finish_speculative_target_commit(True)
        torch.testing.assert_close(self.state_base, committed_state, atol=0, rtol=0)
        torch.testing.assert_close(self.ctx_base, committed_context, atol=0, rtol=0)
        self.model.finish_speculative_target_commit(False)
        torch.testing.assert_close(self.state_base, before_state, atol=0, rtol=0)
        torch.testing.assert_close(self.ctx_base, before_context, atol=0, rtol=0)

    def test_async_budget_fallback_preserves_prefill_pages(self):
        embedding = self._enable_async_cpu_ple()
        embedding._prefetcher._max_eager_bytes = 1
        ids = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.long)
        hyper = torch.randn(4, _HC * _HIDDEN, device="cuda", dtype=torch.bfloat16)
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([3, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        before_state, before_context = self.state_base.clone(), self.ctx_base.clone()
        expected = reference_apply_ple(
            self.model._ple_execution(), 1, hyper, ids, inputs
        )
        expected_state, expected_context = (
            self.state_base.clone(),
            self.ctx_base.clone(),
        )
        self.state_base.copy_(before_state)
        self.ctx_base.copy_(before_context)
        plans = self.model._start_eager_ple_prefetch(
            SimpleNamespace(input_ids=ids), inputs
        )
        self.assertTrue(all(handle is None for handle in plans[1]))
        actual = self.model._apply_ple(1, hyper, ids, inputs, prefetch=plans[1])
        embedding.finish_prefetch()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(self.state_base, expected_state, atol=0, rtol=0)
        torch.testing.assert_close(self.ctx_base, expected_context, atol=0, rtol=0)

    def test_async_multimodal_prefetch_uses_eos_and_forward_finally_releases_slots(
        self,
    ):
        embedding = self._enable_async_cpu_ple()
        ids = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.long)
        mask = torch.tensor([True, False, True, True], device="cuda")
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([3, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        frame = SimpleNamespace(
            input_ids=ids, embedding_inputs=SimpleNamespace(text_tokens_mask=mask)
        )
        plans = self.model._start_eager_ple_prefetch(frame, inputs)
        history = torch.tensor([[7, 1, 7, 3]], device="cuda", dtype=torch.long)
        expected = embedding._gather_torch(embedding.hashed_ids(history, 3).cpu()).to(
            "cuda"
        )
        torch.testing.assert_close(
            embedding.consume_prefetch(plans[1][0]), expected, atol=0, rtol=0
        )
        with patch.object(
            self.model,
            "_forward_with_ple_prefetch",
            side_effect=RuntimeError("before consume"),
        ):
            with self.assertRaisesRegex(RuntimeError, "before consume"):
                self.model.forward(frame)
        self.assertFalse(embedding._prefetcher._active_slots)


if __name__ == "__main__":
    main()
