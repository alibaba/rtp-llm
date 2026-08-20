import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import KimiK3ChunkRound
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import KimiKDACurrentStateRegistry
from rtp_llm.ops.compute_ops import PyAttentionInputs


class KimiK3MtpChunkPrefillUnitTest(unittest.TestCase):
    @staticmethod
    def _model(*, ep_size: int = 1):
        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        object.__setattr__(model, "_is_decode_role", False)
        object.__setattr__(model, "_mtp_hidden_buffer", None)
        object.__setattr__(model, "_mtp_hidden_valid_tokens", 0)
        object.__setattr__(model, "_layer_group_ids", None)
        object.__setattr__(
            model, "kv_cache", SimpleNamespace(seq_size_per_block=64)
        )
        object.__setattr__(
            model,
            "parallelism_config",
            SimpleNamespace(get_attn_tp_size=lambda: 1, ep_size=ep_size),
        )
        object.__setattr__(model, "layers", [])
        return model

    @staticmethod
    def _inputs(num_tokens: int, input_lengths, prefix_lengths):
        attention_inputs = PyAttentionInputs()
        attention_inputs.is_prefill = True
        attention_inputs.input_lengths_host = torch.tensor(
            input_lengths, dtype=torch.int32
        )
        attention_inputs.prefix_lengths_host = torch.tensor(
            prefix_lengths, dtype=torch.int32
        )
        attention_inputs.kv_cache_layer_to_group_host = torch.tensor(
            [0], dtype=torch.int32
        )
        return SimpleNamespace(
            input_ids=torch.arange(num_tokens, dtype=torch.int32),
            attention_inputs=attention_inputs,
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[], mm_features_locs_host=None
            ),
        )

    def test_forward_dispatches_oversized_prefill_to_whole_chunk_path(self) -> None:
        model = self._model()
        whole_chunk = MagicMock(return_value=object())
        impl_one = MagicMock(return_value=object())
        object.__setattr__(model, "_forward_whole_chunk_prefill", whole_chunk)
        object.__setattr__(model, "_forward_impl_one", impl_one)
        chunk_inputs = self._inputs(8, [8], [0])
        small_inputs = self._inputs(4, [4], [0])

        with patch(
            "rtp_llm.models_py.model_desc.kimi_k3.prefill_chunk_tokens",
            return_value=4,
        ):
            result = model.forward(chunk_inputs)
            self.assertIsNotNone(result)
            whole_chunk.assert_called_once_with(chunk_inputs, None, 4, None)
            impl_one.assert_not_called()

            whole_chunk.reset_mock()
            model.forward(small_inputs)
            impl_one.assert_called_once_with(small_inputs, None)
            whole_chunk.assert_not_called()

        impl_one.reset_mock()
        with patch(
            "rtp_llm.models_py.model_desc.kimi_k3.prefill_chunk_tokens",
            return_value=0,
        ):
            model.forward(chunk_inputs)
        impl_one.assert_called_once_with(chunk_inputs, None)
        whole_chunk.assert_not_called()

    def test_whole_chunk_prefill_plans_rounds_hooks_and_collects_terminal_row(
        self,
    ) -> None:
        model = self._model()
        hidden_outputs = [
            SimpleNamespace(
                hidden_states=torch.arange(0, 64, dtype=torch.float32).reshape(64, 1),
                params_ptr=None,
            ),
            SimpleNamespace(
                hidden_states=torch.arange(64, 128, dtype=torch.float32).reshape(64, 1),
                params_ptr=None,
            ),
        ]
        forward_kwargs = []
        hook_calls = []

        def forward_one(round_inputs, _fmha_impl, **_kwargs):
            forward_kwargs.append((round_inputs, _kwargs))
            output = hidden_outputs.pop(0)
            model._mtp_hidden_buffer = output.hidden_states + 1000
            model._mtp_hidden_valid_tokens = output.hidden_states.size(0)
            return output

        def record_hook(round_plan, is_last):
            hook_calls.append((round_plan, is_last))

        release_mock = MagicMock()
        forward_mock = MagicMock(side_effect=forward_one)
        publish_mock = MagicMock()
        ordered = MagicMock()
        ordered.attach_mock(release_mock, "release")
        ordered.attach_mock(forward_mock, "forward")
        object.__setattr__(model, "_forward_impl_one", forward_mock)
        object.__setattr__(
            model, "_release_prefill_mtp_hidden_buffer", release_mock
        )
        object.__setattr__(model, "_publish_whole_chunk_cache", publish_mock)

        inputs = self._inputs(128, [128], [0])
        fmha = MagicMock()
        with patch("rtp_llm.models_py.model_desc.kimi_k3.barrier") as barrier:
            result = model._forward_whole_chunk_prefill(
                inputs, fmha, 64, record_hook
            )

        barrier.assert_called_once()
        self.assertEqual(
            [entry[0] for entry in ordered.mock_calls],
            ["release", "forward", "release", "forward"],
        )
        self.assertEqual(len(hook_calls), 2)
        plan0, is_last0 = hook_calls[0]
        plan1, is_last1 = hook_calls[1]
        self.assertIsInstance(plan0, KimiK3ChunkRound)
        self.assertFalse(is_last0)
        self.assertTrue(is_last1)
        self.assertFalse(plan0.slices[0].terminal)
        self.assertTrue(plan1.slices[0].terminal)
        self.assertEqual(plan0.token_count, 64)
        self.assertEqual(plan1.token_count, 64)
        self.assertIs(forward_kwargs[0][1]["round_plan"], plan0)
        self.assertIs(forward_kwargs[1][1]["round_plan"], plan1)
        registry0 = forward_kwargs[0][1]["kda_current_state_registry"]
        self.assertIsInstance(registry0, KimiKDACurrentStateRegistry)
        self.assertIs(forward_kwargs[1][1]["kda_current_state_registry"], registry0)
        self.assertEqual(
            forward_kwargs[0][0].input_ids.tolist(), list(range(0, 64))
        )
        self.assertEqual(
            forward_kwargs[1][0].input_ids.tolist(), list(range(64, 128))
        )
        self.assertEqual(fmha.prepare.call_count, 2)
        for round_inputs, _ in forward_kwargs:
            self.assertTrue(round_inputs.attention_inputs.is_prefill)
            self.assertIsNone(round_inputs.attention_inputs.cache_store_inputs)
        publish_mock.assert_called_once_with(inputs.attention_inputs)
        self.assertTrue(result.lm_output_already_selected)
        self.assertTrue(torch.equal(result.hidden_states, torch.tensor([[127.0]])))
        self.assertTrue(
            torch.equal(model._mtp_hidden_buffer, torch.tensor([[1127.0]]))
        )

    def test_whole_chunk_prefill_collects_one_terminal_row_per_request(self) -> None:
        model = self._model()
        hidden_outputs = [
            SimpleNamespace(
                hidden_states=torch.tensor(
                    [[float(i)] for i in range(0, 64)]
                ),
                params_ptr=None,
            ),
            SimpleNamespace(
                hidden_states=torch.tensor(
                    [[float(i)] for i in range(64, 128)]
                ),
                params_ptr=None,
            ),
        ]
        is_last_flags = []

        def forward_one(*_args, **_kwargs):
            output = hidden_outputs.pop(0)
            model._mtp_hidden_buffer = output.hidden_states + 1000
            model._mtp_hidden_valid_tokens = output.hidden_states.size(0)
            return output

        def record_hook(_round_plan, is_last):
            is_last_flags.append(is_last)

        object.__setattr__(
            model, "_forward_impl_one", MagicMock(side_effect=forward_one)
        )
        object.__setattr__(model, "_publish_whole_chunk_cache", MagicMock())
        inputs = self._inputs(128, [64, 64], [0, 0])
        with patch("rtp_llm.models_py.model_desc.kimi_k3.barrier"):
            result = model._forward_whole_chunk_prefill(
                inputs, MagicMock(), 64, record_hook
            )

        self.assertEqual(is_last_flags, [False, True])
        self.assertTrue(result.lm_output_already_selected)
        self.assertTrue(
            torch.equal(result.hidden_states, torch.tensor([[63.0], [127.0]]))
        )
        self.assertTrue(
            torch.equal(
                model._mtp_hidden_buffer, torch.tensor([[1063.0], [1127.0]])
            )
        )

    def test_each_round_releases_stale_mtp_buffer_before_target_forward(
        self,
    ) -> None:
        model = self._model()
        model._mtp_hidden_buffer = torch.ones((4, 8))
        model._mtp_hidden_valid_tokens = 4
        observed = []

        def forward_one(*_args, **_kwargs):
            observed.append(
                (model._mtp_hidden_buffer, model._mtp_hidden_valid_tokens)
            )
            return SimpleNamespace(
                hidden_states=torch.zeros((64, 2)), params_ptr=None
            )

        object.__setattr__(
            model, "_forward_impl_one", MagicMock(side_effect=forward_one)
        )
        object.__setattr__(model, "_publish_whole_chunk_cache", MagicMock())
        with patch("rtp_llm.models_py.model_desc.kimi_k3.barrier"):
            model._forward_whole_chunk_prefill(
                self._inputs(128, [128], [0]), MagicMock(), 64
            )

        self.assertEqual(len(observed), 2)
        self.assertTrue(
            all(buffer is None and tokens == 0 for buffer, tokens in observed)
        )

    def test_prefill_hidden_release_keeps_decode_graph_buffer(self) -> None:
        model = self._model()
        model._is_decode_role = True
        graph_buffer = torch.ones((4, 8))
        model._mtp_hidden_buffer = graph_buffer
        model._mtp_hidden_valid_tokens = 4

        model._release_prefill_mtp_hidden_buffer()

        self.assertIs(model._mtp_hidden_buffer, graph_buffer)
        self.assertEqual(model._mtp_hidden_valid_tokens, 4)

    def test_prefill_hidden_release_flushes_allocator_cache(self) -> None:
        model = self._model()
        model._mtp_hidden_buffer = torch.ones((4, 8))
        model._mtp_hidden_valid_tokens = 4

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            model._release_prefill_mtp_hidden_buffer()

        self.assertIsNone(model._mtp_hidden_buffer)
        self.assertEqual(model._mtp_hidden_valid_tokens, 0)
        empty_cache.assert_called_once_with()

    def test_prefill_hidden_release_without_buffer_does_not_flush(self) -> None:
        model = self._model()

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            model._release_prefill_mtp_hidden_buffer()

        empty_cache.assert_not_called()

    def test_abort_prefill_chunk_session_releases_buffer(self) -> None:
        model = self._model()
        model._mtp_hidden_buffer = torch.ones((4, 8))
        model._mtp_hidden_valid_tokens = 4

        model.abort_prefill_chunk_session()

        self.assertIsNone(model._mtp_hidden_buffer)
        self.assertEqual(model._mtp_hidden_valid_tokens, 0)

    def test_round_failure_propagates(self) -> None:
        model = self._model()
        object.__setattr__(
            model,
            "_forward_impl_one",
            MagicMock(side_effect=RuntimeError("injected target failure")),
        )

        with (
            patch("rtp_llm.models_py.model_desc.kimi_k3.barrier"),
            self.assertRaisesRegex(RuntimeError, "injected target failure"),
        ):
            model._forward_whole_chunk_prefill(
                self._inputs(128, [128], [0]), MagicMock(), 64
            )

    def test_whole_chunk_prefill_requires_host_layer_group_map(self) -> None:
        model = self._model()
        inputs = self._inputs(8, [8], [0])
        attention_inputs = PyAttentionInputs()
        attention_inputs.is_prefill = True
        attention_inputs.input_lengths_host = torch.tensor([8], dtype=torch.int32)
        attention_inputs.prefix_lengths_host = torch.tensor([0], dtype=torch.int32)
        inputs.attention_inputs = attention_inputs

        with patch(
            "rtp_llm.models_py.model_desc.kimi_k3.barrier"
        ) as barrier:
            with self.assertRaisesRegex(
                RuntimeError, "requires a host layer/group map"
            ):
                model._forward_whole_chunk_prefill(inputs, MagicMock(), 4)

        barrier.assert_not_called()

    def test_whole_chunk_prefill_rejects_tp_ep_mismatch(self) -> None:
        model = self._model(ep_size=2)

        with self.assertRaisesRegex(RuntimeError, "TP == EP"):
            model._forward_whole_chunk_prefill(
                self._inputs(8, [8], [0]), MagicMock(), 4
            )

    def test_whole_chunk_prefill_rejects_lengths_not_covering_input(self) -> None:
        model = self._model()

        with self.assertRaisesRegex(RuntimeError, "do not cover input tokens"):
            model._forward_whole_chunk_prefill(
                self._inputs(8, [4], [0]), MagicMock(), 4
            )


if __name__ == "__main__":
    unittest.main()
