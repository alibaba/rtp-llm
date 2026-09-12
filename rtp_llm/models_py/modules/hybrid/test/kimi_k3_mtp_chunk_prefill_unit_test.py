import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import (
    KimiK3ChunkRound,
    KimiK3ChunkSlice,
    build_chunk_model_inputs,
)
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import KimiKDACurrentStateRegistry
from rtp_llm.ops.compute_ops import CacheGroupType, PyAttentionInputs


class KimiK3MtpChunkPrefillUnitTest(unittest.TestCase):
    @staticmethod
    def _model(*, ep_size: int = 1):
        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        object.__setattr__(model, "_is_decode_role", False)
        object.__setattr__(model, "_mtp_hidden_buffer", None)
        object.__setattr__(model, "_mtp_hidden_valid_tokens", 0)
        object.__setattr__(model, "_prefill_mtp_hidden_workspace", None)
        object.__setattr__(model, "_prefill_mtp_draft_workspace", None)
        object.__setattr__(model, "_whole_chunk_prefill_active", False)
        object.__setattr__(model, "_layer_group_ids", None)
        object.__setattr__(model, "_k3_page_tokens", 64)
        object.__setattr__(model, "_kda_checkpoint_tokens", 64)
        object.__setattr__(model, "config", SimpleNamespace(hidden_size=4))
        object.__setattr__(model, "embedding_weight", torch.empty((1, 4)))
        object.__setattr__(
            model,
            "kv_cache",
            SimpleNamespace(seq_size_per_block=64, local_shard_count=1),
        )
        object.__setattr__(
            model,
            "parallelism_config",
            SimpleNamespace(
                get_attn_tp_size=lambda: 1,
                kv_page_rr_enabled=lambda: False,
                ep_size=ep_size,
            ),
        )
        object.__setattr__(model, "layers", [])
        return model

    def test_cacheless_warmup_binds_checkpoint_once_when_cache_arrives(self):
        model = self._model()
        object.__setattr__(model, "kv_cache", None)
        object.__setattr__(model, "_kda_checkpoint_tokens", None)
        object.__setattr__(
            model,
            "layers",
            [SimpleNamespace(is_kda=False), SimpleNamespace(is_kda=True)],
        )
        object.__setattr__(
            model,
            "parallelism_config",
            SimpleNamespace(
                tp_size=8,
                tp_rank=3,
                kv_page_rr_enabled=lambda: True,
                prefill_cp_config=SimpleNamespace(prefill_cp_size=8),
            ),
        )
        model._initialize_k3_cache_geometry()
        self.assertIsNone(model._kda_checkpoint_tokens)
        object.__setattr__(
            model,
            "kv_cache",
            SimpleNamespace(
                seq_size_per_block=128,
                local_shard_count=8,
                group_seq_size_per_block=[128, 1024],
                layer_group_types=[CacheGroupType.FULL, CacheGroupType.LINEAR],
                get_layer_cache=lambda layer: SimpleNamespace(group_id=layer),
            ),
        )
        model._initialize_k3_cache_geometry()
        self.assertEqual(
            (model._k3_page_tokens, model._kda_checkpoint_tokens), (128, 1024)
        )
        # Later mutation must not silently rebind the service's planning units.
        model.kv_cache.group_seq_size_per_block = [128, 128]
        model._initialize_k3_cache_geometry()
        self.assertEqual(
            (model._k3_page_tokens, model._kda_checkpoint_tokens), (128, 1024)
        )

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
                multimodal_features=[],
                mm_features_locs_host=None,
                mm_extra_input=[],
            ),
            embedding_inputs=SimpleNamespace(
                combo_tokens_type_ids=None,
                text_tokens_mask=None,
            ),
            force_disable_sp_run=False,
        )

    def test_chunk_round_recomputes_logical_and_physical_padding_counts(self):
        inputs = self._inputs(8, [5, 3], [0, 0])
        inputs.attention_inputs.logical_request_count = 1
        inputs.attention_inputs.coordinated_request_count = 1
        inputs.attention_inputs.physical_request_count = 2
        inputs.attention_inputs.logical_token_count = 5
        inputs.attention_inputs.coordinated_token_count = 5
        inputs.attention_inputs.physical_token_count = 8
        round_plan = KimiK3ChunkRound(
            (
                KimiK3ChunkSlice(0, 0, 5, 0, 0, 5, 0, 5, True),
                KimiK3ChunkSlice(1, 5, 8, 0, 0, 3, 0, 3, True),
            )
        )

        chunk = build_chunk_model_inputs(
            inputs.input_ids,
            inputs.attention_inputs,
            round_plan=round_plan,
        )

        attention = chunk.attention_inputs
        self.assertEqual(attention.logical_request_count, 1)
        self.assertEqual(attention.coordinated_request_count, 1)
        self.assertEqual(attention.physical_request_count, 2)
        self.assertEqual(attention.logical_token_count, 5)
        self.assertEqual(attention.coordinated_token_count, 5)
        self.assertEqual(attention.physical_token_count, 8)
        self.assertTrue(attention.is_s_padded)

    def test_chunk_inputs_slice_multimodal_rows_without_cross_request_leakage(
        self,
    ) -> None:
        inputs = self._inputs(8, [4, 4], [0, 0])
        first_feature = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        # The second feature belongs to request 1. Its first row is already in
        # that request's reused prefix, so its gathered location falls at the
        # final packed position of request 0.
        second_feature = torch.arange(12, 24, dtype=torch.float32).reshape(3, 4)
        inputs.multimodal_inputs = SimpleNamespace(
            multimodal_features=[first_feature, second_feature],
            mm_features_locs_host=torch.tensor([2, 3], dtype=torch.int32),
            mm_extra_input=[],
        )
        inputs.embedding_inputs.text_tokens_mask = torch.tensor(
            [1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.bool
        )

        first_round = KimiK3ChunkRound(
            slices=(
                KimiK3ChunkSlice(0, 0, 3, 0, 0, 3, 0, 3, False),
                KimiK3ChunkSlice(1, 4, 5, 0, 0, 1, 0, 1, False),
            )
        )
        second_round = KimiK3ChunkRound(
            slices=(
                KimiK3ChunkSlice(0, 3, 4, 0, 3, 1, 3, 4, True),
                KimiK3ChunkSlice(1, 5, 8, 0, 1, 3, 1, 4, True),
            )
        )

        first = build_chunk_model_inputs(
            inputs.input_ids,
            inputs.attention_inputs,
            round_plan=first_round,
            multimodal_inputs=inputs.multimodal_inputs,
            embedding_inputs=inputs.embedding_inputs,
        )
        second = build_chunk_model_inputs(
            inputs.input_ids,
            inputs.attention_inputs,
            round_plan=second_round,
            multimodal_inputs=inputs.multimodal_inputs,
            embedding_inputs=inputs.embedding_inputs,
        )

        self.assertEqual(first.input_ids.tolist(), [0, 1, 2, 4])
        self.assertEqual(first.multimodal_inputs.mm_features_locs_host.tolist(), [2, 3])
        torch.testing.assert_close(
            first.multimodal_inputs.multimodal_features[0], first_feature[:1]
        )
        torch.testing.assert_close(
            first.multimodal_inputs.multimodal_features[1], second_feature[1:2]
        )
        self.assertEqual(
            first.embedding_inputs.text_tokens_mask.tolist(), [True, True, False, False]
        )

        self.assertEqual(second.input_ids.tolist(), [3, 5, 6, 7])
        self.assertEqual(
            second.multimodal_inputs.mm_features_locs_host.tolist(), [0, 1]
        )
        torch.testing.assert_close(
            second.multimodal_inputs.multimodal_features[0], first_feature[1:]
        )
        torch.testing.assert_close(
            second.multimodal_inputs.multimodal_features[1], second_feature[2:]
        )
        self.assertEqual(
            second.embedding_inputs.text_tokens_mask.tolist(), [True, False, True, True]
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_chunk_metadata_uses_pinned_host_staging_for_async_h2d(self) -> None:
        inputs = self._inputs(4, [2, 2], [0, 0])
        inputs.input_ids = inputs.input_ids.cuda()
        inputs.attention_inputs.kv_cache_block_id_host = torch.arange(
            8, dtype=torch.int32
        ).reshape(2, 4)
        inputs.attention_inputs.kv_cache_kernel_block_id_device = torch.arange(
            8, dtype=torch.int32, device="cuda"
        ).reshape(2, 4)
        plan = KimiK3ChunkRound(
            (
                KimiK3ChunkSlice(1, 2, 4, 0, 0, 2, 0, 2, True),
            )
        )

        chunk = build_chunk_model_inputs(
            inputs.input_ids,
            inputs.attention_inputs,
            round_plan=plan,
        )
        torch.cuda.synchronize()

        self.assertTrue(chunk.attention_inputs.input_lengths_host.is_pinned())
        self.assertTrue(chunk.attention_inputs.prefix_lengths_host.is_pinned())
        self.assertEqual(chunk.attention_inputs.input_lengths.tolist(), [2])
        self.assertEqual(
            chunk.attention_inputs.kv_cache_block_id_host.tolist(), [[4, 5, 6, 7]]
        )
        self.assertEqual(
            chunk.attention_inputs.kv_cache_kernel_block_id_device.tolist(),
            [[4, 5, 6, 7]],
        )

    def test_multimodal_chunk_clips_features_and_repacks_locations(self):
        inputs = self._inputs(8, [4, 4], [5, 0])
        a = torch.arange(20).reshape(5, 4)
        b = torch.arange(12).reshape(3, 4) + 100
        mm = SimpleNamespace(
            multimodal_features=[a, b],
            mm_features_locs_host=torch.tensor([-1, 4]),
        )
        plan = KimiK3ChunkRound(
            (
                KimiK3ChunkSlice(0, 2, 4, 5, 2, 2, 7, 9, True),
                KimiK3ChunkSlice(1, 4, 6, 0, 0, 2, 0, 2, False),
            )
        )
        chunk = build_chunk_model_inputs(
            inputs.input_ids,
            inputs.attention_inputs,
            round_plan=plan,
            multimodal_inputs=mm,
        )
        self.assertEqual(chunk.input_ids.tolist(), [2, 3, 4, 5])
        self.assertEqual(chunk.multimodal_inputs.mm_features_locs_host.tolist(), [0, 2])
        actual = chunk.multimodal_inputs.multimodal_features
        torch.testing.assert_close(actual[0], a[3:5])
        torch.testing.assert_close(actual[1], b[:2])
        self.assertEqual(mm.mm_features_locs_host.tolist(), [-1, 4])

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
            self.assertFalse(model._whole_chunk_prefill_active)
            self.assertIsNone(model._prefill_mtp_hidden_workspace)
            self.assertIsNone(model._prefill_mtp_draft_workspace)

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
            ordered.draft()
            hook_calls.append((round_plan, is_last))

        release_mock = MagicMock()
        forward_mock = MagicMock(side_effect=forward_one)
        publish_mock = MagicMock()
        ordered = MagicMock()
        ordered.attach_mock(release_mock, "release")
        ordered.attach_mock(forward_mock, "forward")
        object.__setattr__(model, "_forward_impl_one", forward_mock)
        object.__setattr__(model, "_release_prefill_mtp_hidden_buffer", release_mock)
        object.__setattr__(model, "_publish_whole_chunk_cache", publish_mock)

        inputs = self._inputs(128, [128], [0])
        feature = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        inputs.multimodal_inputs = SimpleNamespace(
            multimodal_features=[feature],
            mm_features_locs_host=torch.tensor([62], dtype=torch.int32),
            mm_extra_input=[],
        )
        inputs.embedding_inputs.text_tokens_mask = torch.ones(128, dtype=torch.bool)
        inputs.embedding_inputs.text_tokens_mask[62:66] = False
        fmha = MagicMock()
        ordered.attach_mock(fmha.release_forward_workspace, "attention_workspace")
        with patch("rtp_llm.models_py.model_desc.kimi_k3.barrier") as barrier:
            result = model._forward_whole_chunk_prefill(inputs, fmha, 64, record_hook)

        barrier.assert_called_once()
        self.assertEqual(
            [entry[0] for entry in ordered.mock_calls],
            ["release", "forward", "attention_workspace", "draft"] * 2,
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
        self.assertEqual(forward_kwargs[0][0].input_ids.tolist(), list(range(0, 64)))
        self.assertEqual(forward_kwargs[1][0].input_ids.tolist(), list(range(64, 128)))
        first_mm = forward_kwargs[0][0].multimodal_inputs
        second_mm = forward_kwargs[1][0].multimodal_inputs
        self.assertEqual(first_mm.mm_features_locs_host.tolist(), [62])
        self.assertEqual(second_mm.mm_features_locs_host.tolist(), [0])
        torch.testing.assert_close(first_mm.multimodal_features[0], feature[:2])
        torch.testing.assert_close(second_mm.multimodal_features[0], feature[2:])
        self.assertEqual(
            int(forward_kwargs[0][0].embedding_inputs.text_tokens_mask.sum()), 62
        )
        self.assertEqual(
            int(forward_kwargs[1][0].embedding_inputs.text_tokens_mask.sum()), 62
        )
        self.assertEqual(fmha.prepare.call_count, 2)
        for round_inputs, _ in forward_kwargs:
            self.assertTrue(round_inputs.attention_inputs.is_prefill)
            self.assertIsNone(round_inputs.attention_inputs.cache_store_inputs)
        publish_mock.assert_called_once_with(inputs.attention_inputs)
        self.assertTrue(result.lm_output_already_selected)
        self.assertTrue(torch.equal(result.hidden_states, torch.tensor([[127.0]])))
        self.assertTrue(torch.equal(model._mtp_hidden_buffer, torch.tensor([[1127.0]])))

    def test_whole_chunk_prefill_collects_one_terminal_row_per_request(self) -> None:
        model = self._model()
        hidden_outputs = [
            SimpleNamespace(
                hidden_states=torch.tensor([[float(i)] for i in range(0, 64)]),
                params_ptr=None,
            ),
            SimpleNamespace(
                hidden_states=torch.tensor([[float(i)] for i in range(64, 128)]),
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
            torch.equal(model._mtp_hidden_buffer, torch.tensor([[1063.0], [1127.0]]))
        )

    def test_target_only_chunk_prefill_propagates_disable_flag_without_staging_mtp_hidden(
        self,
    ) -> None:
        model = self._model()
        hook = MagicMock()

        def forward_one(round_inputs, *_args, **_kwargs):
            self.assertTrue(round_inputs.force_disable_sp_run)
            self.assertIsNone(model._mtp_hidden_buffer)
            return SimpleNamespace(
                hidden_states=round_inputs.input_ids.float().reshape(-1, 1),
                params_ptr=None,
            )

        object.__setattr__(
            model, "_forward_impl_one", MagicMock(side_effect=forward_one)
        )
        object.__setattr__(model, "_publish_whole_chunk_cache", MagicMock())
        inputs = self._inputs(128, [128], [0])
        inputs.force_disable_sp_run = True

        with patch("rtp_llm.models_py.model_desc.kimi_k3.barrier"):
            result = model._forward_whole_chunk_prefill(inputs, MagicMock(), 64, hook)

        self.assertEqual(hook.call_count, 2)
        self.assertIsNone(model._mtp_hidden_buffer)
        self.assertEqual(model._mtp_hidden_valid_tokens, 0)
        self.assertTrue(torch.equal(result.hidden_states, torch.tensor([[127.0]])))

    def test_each_round_releases_stale_mtp_buffer_before_target_forward(
        self,
    ) -> None:
        model = self._model()
        model._mtp_hidden_buffer = torch.ones((4, 8))
        model._mtp_hidden_valid_tokens = 4
        observed = []

        def forward_one(*_args, **_kwargs):
            observed.append((model._mtp_hidden_buffer, model._mtp_hidden_valid_tokens))
            return SimpleNamespace(hidden_states=torch.zeros((64, 2)), params_ptr=None)

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

    def test_prefill_hidden_release_keeps_workspaces_and_never_flushes_cache(
        self,
    ) -> None:
        model = self._model()
        hidden_workspace = torch.ones((4, 8))
        draft_workspace = torch.zeros((4, 8))
        model._prefill_mtp_hidden_workspace = hidden_workspace
        model._prefill_mtp_draft_workspace = draft_workspace
        model._mtp_hidden_buffer = hidden_workspace
        model._mtp_hidden_valid_tokens = 4

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            model._release_prefill_mtp_hidden_buffer()

        self.assertIsNone(model._mtp_hidden_buffer)
        self.assertEqual(model._mtp_hidden_valid_tokens, 0)
        self.assertIs(model._prefill_mtp_hidden_workspace, hidden_workspace)
        self.assertIs(model._prefill_mtp_draft_workspace, draft_workspace)
        empty_cache.assert_not_called()

    def test_prefill_hidden_release_without_buffer_does_not_flush(self) -> None:
        model = self._model()

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "empty_cache") as empty_cache,
        ):
            model._release_prefill_mtp_hidden_buffer()

        empty_cache.assert_not_called()

    def test_prefill_hidden_write_and_terminal_compaction_reuse_fixed_storage(
        self,
    ) -> None:
        model = self._model()
        hidden_workspace = torch.empty((8, 4))
        draft_workspace = torch.empty((8, 4))
        model._prefill_mtp_hidden_workspace = hidden_workspace
        model._prefill_mtp_draft_workspace = draft_workspace
        model._whole_chunk_prefill_active = True
        hidden_ptr = hidden_workspace.data_ptr()
        draft_ptr = draft_workspace.data_ptr()

        first = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        model._write_whole_chunk_mtp_hidden(first)
        self.assertEqual(model._mtp_hidden_buffer.data_ptr(), hidden_ptr)
        torch.testing.assert_close(model._mtp_hidden_buffer, first)

        index = [0, 1, 3, 4]
        selected = model._select_prefill_mtp_draft_rows(
            model._mtp_hidden_buffer,
            index,
        )
        self.assertEqual(selected.data_ptr(), draft_ptr)
        torch.testing.assert_close(
            selected,
            first.index_select(0, torch.tensor(index, dtype=torch.long)),
        )

        model._release_prefill_mtp_hidden_buffer()
        second = first + 100
        model._write_whole_chunk_mtp_hidden(second)
        self.assertEqual(model._mtp_hidden_buffer.data_ptr(), hidden_ptr)
        torch.testing.assert_close(model._mtp_hidden_buffer, second)

    def test_non_chunk_prefill_keeps_original_hidden_tensor(self) -> None:
        model = self._model()
        hidden_workspace = torch.empty((8, 4))
        model._prefill_mtp_hidden_workspace = hidden_workspace
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)

        model._write_mtp_hidden_buffer(hidden, is_cuda_graph=False)

        self.assertIs(model._mtp_hidden_buffer, hidden)
        self.assertEqual(model._mtp_hidden_valid_tokens, 6)
        self.assertIs(model._prefill_mtp_hidden_workspace, hidden_workspace)

    def test_prefill_draft_rows_without_terminal_are_zero_copy(self) -> None:
        model = self._model()
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        model._prefill_mtp_draft_workspace = torch.empty_like(hidden)
        index = list(range(6))

        selected = model._select_prefill_mtp_draft_rows(hidden, index)

        self.assertIs(selected, hidden)

    def test_prefill_terminal_prefix_reuses_hidden_workspace_view(self) -> None:
        model = self._model()
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)

        selected = model._select_prefill_mtp_draft_rows(hidden, list(range(5)))

        self.assertEqual(selected.data_ptr(), hidden.data_ptr())
        self.assertIsNone(model._prefill_mtp_draft_workspace)
        torch.testing.assert_close(selected, hidden[:5])

    def test_prefill_nonprefix_compaction_allocates_workspace_lazily(self) -> None:
        model = self._model()
        hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        model._prefill_mtp_hidden_workspace = torch.empty((8, 4))
        index = [0, 2, 3, 5]

        selected = model._select_prefill_mtp_draft_rows(hidden, index)

        self.assertIsNotNone(model._prefill_mtp_draft_workspace)
        self.assertEqual(model._prefill_mtp_draft_workspace.shape, (8, 4))
        self.assertEqual(
            selected.data_ptr(), model._prefill_mtp_draft_workspace.data_ptr()
        )
        torch.testing.assert_close(
            selected,
            hidden.index_select(0, torch.tensor(index, dtype=torch.long)),
        )

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

        fmha = MagicMock()
        with (
            patch("rtp_llm.models_py.model_desc.kimi_k3.barrier"),
            self.assertRaisesRegex(RuntimeError, "injected target failure"),
        ):
            model._forward_whole_chunk_prefill(self._inputs(128, [128], [0]), fmha, 64)

        self.assertFalse(model._whole_chunk_prefill_active)
        self.assertIsNone(model._prefill_mtp_hidden_workspace)
        self.assertIsNone(model._prefill_mtp_draft_workspace)

    def test_nested_chunk_prefill_does_not_clear_outer_session(self) -> None:
        model = self._model()
        workspace = torch.empty((8, 4))
        model._whole_chunk_prefill_active = True
        model._prefill_mtp_hidden_workspace = workspace
        fmha = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "nested"):
            model._forward_whole_chunk_prefill(self._inputs(8, [8], [0]), fmha, 4)

        self.assertTrue(model._whole_chunk_prefill_active)
        self.assertIs(model._prefill_mtp_hidden_workspace, workspace)

    def test_whole_chunk_prefill_requires_host_layer_group_map(self) -> None:
        model = self._model()
        inputs = self._inputs(8, [8], [0])
        attention_inputs = PyAttentionInputs()
        attention_inputs.is_prefill = True
        attention_inputs.input_lengths_host = torch.tensor([8], dtype=torch.int32)
        attention_inputs.prefix_lengths_host = torch.tensor([0], dtype=torch.int32)
        inputs.attention_inputs = attention_inputs

        with patch("rtp_llm.models_py.model_desc.kimi_k3.barrier") as barrier:
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
