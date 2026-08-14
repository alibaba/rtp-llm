from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.model_desc.kimi_k3_chunk_planner import (
    KimiK3ChunkRound,
    KimiK3ChunkSlice,
)
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyContextParallelParams,
    PyModelInputs,
)


class KimiK3ChunkMetadataTest(unittest.TestCase):
    @staticmethod
    def _validation_model() -> KimiK3Model:
        model = object.__new__(KimiK3Model)
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 8,
            ep_size=8,
        )
        model.kv_cache = SimpleNamespace(seq_size_per_block=64)
        return model

    @staticmethod
    def _validation_inputs() -> PyModelInputs:
        inputs = PyModelInputs()
        inputs.input_ids = torch.arange(512, dtype=torch.int32)
        inputs.attention_inputs = PyAttentionInputs()
        return inputs

    @staticmethod
    def _round() -> KimiK3ChunkRound:
        return KimiK3ChunkRound(
            (
                KimiK3ChunkSlice(
                    original_batch_idx=2,
                    source_start=30,
                    source_end=35,
                    prefix_length=64,
                    processed_length=0,
                    new_length=5,
                    absolute_start=64,
                    absolute_end=69,
                    terminal=True,
                ),
                KimiK3ChunkSlice(
                    original_batch_idx=0,
                    source_start=0,
                    source_end=3,
                    prefix_length=128,
                    processed_length=0,
                    new_length=3,
                    absolute_start=128,
                    absolute_end=131,
                    terminal=True,
                ),
            )
        )

    def test_undefined_cpp_tensor_fields_remain_undefined(self) -> None:
        source = PyAttentionInputs()
        source.kv_cache_block_id_host = torch.arange(12, dtype=torch.int32).view(
            3, 4
        )
        source.kv_cache_kernel_block_id_host = (
            torch.arange(12, dtype=torch.int32).view(3, 4) + 100
        )

        chunk = KimiK3Model._chunk_attention_inputs(
            source,
            round_plan=self._round(),
            device=torch.device("cpu"),
        )

        self.assertIsNone(chunk.kv_cache_block_id_device)
        self.assertIsNone(chunk.kv_cache_kernel_block_id_device)
        torch.testing.assert_close(
            chunk.kv_cache_block_id_host,
            source.kv_cache_block_id_host.index_select(0, torch.tensor([2, 0])),
        )
        torch.testing.assert_close(
            chunk.kv_cache_kernel_block_id_host,
            source.kv_cache_kernel_block_id_host.index_select(
                0, torch.tensor([2, 0])
            ),
        )

    def test_round_lengths_positions_and_original_indices_are_rebuilt(self) -> None:
        source = PyAttentionInputs()
        chunk = KimiK3Model._chunk_attention_inputs(
            source,
            round_plan=self._round(),
            device=torch.device("cpu"),
        )

        self.assertEqual(chunk.cu_seqlens.tolist(), [0, 5, 8])
        self.assertEqual(chunk.cu_kv_seqlens.tolist(), [0, 69, 200])
        self.assertEqual(chunk.input_lengths.tolist(), [5, 3])
        self.assertEqual(chunk.prefix_lengths.tolist(), [64, 128])
        self.assertEqual(chunk.sequence_lengths.tolist(), [69, 131])
        self.assertEqual(chunk.original_batch_indices_host.tolist(), [2, 0])
        self.assertEqual(chunk.padding_offset.tolist(), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(chunk.total_tokens, 8)
        self.assertEqual(chunk.context_total_kv_length, 200)

    def test_round_padding_offset_uses_active_lengths(self) -> None:
        round_plan = KimiK3ChunkRound(
            (
                KimiK3ChunkSlice(
                    original_batch_idx=1,
                    source_start=5,
                    source_end=8,
                    prefix_length=64,
                    processed_length=0,
                    new_length=3,
                    absolute_start=64,
                    absolute_end=67,
                    terminal=True,
                ),
                KimiK3ChunkSlice(
                    original_batch_idx=0,
                    source_start=0,
                    source_end=5,
                    prefix_length=128,
                    processed_length=0,
                    new_length=5,
                    absolute_start=128,
                    absolute_end=133,
                    terminal=True,
                ),
            )
        )
        chunk = KimiK3Model._chunk_attention_inputs(
            PyAttentionInputs(),
            round_plan=round_plan,
            device=torch.device("cpu"),
        )

        self.assertEqual(chunk.cu_seqlens.tolist(), [0, 3, 8])
        self.assertEqual(chunk.padding_offset.tolist(), [0, 0, 0, 2, 2, 2, 2, 2])

    def test_whole_chunk_rejects_unsupported_attention_modes(self) -> None:
        model = self._validation_model()
        for field, message in (
            ("is_target_verify", "target verify"),
            ("is_cuda_graph", "CUDA Graph"),
            ("need_all_logits", "all logits"),
            ("need_all_hidden_states", "all hidden states"),
        ):
            with self.subTest(field=field):
                inputs = self._validation_inputs()
                setattr(inputs.attention_inputs, field, True)
                with self.assertRaisesRegex(RuntimeError, message):
                    model._validate_whole_chunk_prefill(inputs, 256)

    def test_whole_chunk_rejects_context_parallel_and_eagle3(self) -> None:
        model = self._validation_model()
        inputs = self._validation_inputs()
        inputs.attention_inputs.context_parallel_info = PyContextParallelParams()
        with self.assertRaisesRegex(RuntimeError, "Prefill CP"):
            model._validate_whole_chunk_prefill(inputs, 256)

        inputs = self._validation_inputs()
        with mock.patch.dict(os.environ, {"SP_TYPE": "eagle3"}):
            with self.assertRaisesRegex(RuntimeError, "EAGLE3/MTP"):
                model._validate_whole_chunk_prefill(inputs, 256)

    def test_whole_chunk_rejects_multimodal_input(self) -> None:
        model = self._validation_model()
        inputs = self._validation_inputs()
        inputs.multimodal_inputs.mm_features_locs_host = torch.tensor(
            [0], dtype=torch.int32
        )
        with self.assertRaisesRegex(RuntimeError, "multimodal"):
            model._validate_whole_chunk_prefill(inputs, 256)


if __name__ == "__main__":
    unittest.main()
