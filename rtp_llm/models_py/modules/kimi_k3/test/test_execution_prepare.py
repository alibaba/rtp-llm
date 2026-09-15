"""CPU contracts for K3 planning, input ownership and the compute boundary."""

import gc
import os
import weakref
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import (
    KimiK3BatchPlan,
    build_chunk_attention_inputs,
    plan_kimi_k3_chunk_rounds,
)
from rtp_llm.models_py.modules.kimi_k3.input_preparation import (
    KimiK3ExecutionSpec,
    prepare_draft_round,
    prepare_embedding_ids,
    prepare_round,
)
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import plan_kimi_kda_prefill_metadata
from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLA
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs
from torch import nn


class ExecutionPrepareTest(unittest.TestCase):
    def test_draft_selection_matches_packed_reference(self):
        rng = random.Random(41)
        for _ in range(100):
            lengths = [rng.randint(1, 250) for _ in range(rng.randint(1, 8))]
            rounds = plan_kimi_k3_chunk_rounds(
                lengths, [0] * len(lengths), chunk_budget=128, alignment_tokens=64
            )
            batch = KimiK3BatchPlan.from_rounds(rounds)
            seen_terminal = []
            for i, round_plan in enumerate(rounds):
                hidden = torch.arange(round_plan.token_count * 3).reshape(-1, 3)
                expected, offset = [], 0
                for item in round_plan.slices:
                    expected.extend(
                        range(offset, offset + item.new_length - int(item.terminal))
                    )
                    offset += item.new_length
                selection = batch.draft_rows[i]
                selection = selection.prepare(hidden.device)
                torch.testing.assert_close(selection.select(hidden), hidden[expected])
                self.assertEqual(selection.token_count, len(expected))
                if len(selection.ranges) <= 1:
                    self.assertIsNone(selection.index)
                seen_terminal.extend(request for request, _ in batch.terminal_rows[i])
            self.assertEqual(sorted(seen_terminal), list(range(len(lengths))))

    def test_contiguous_selection_is_a_view(self):
        rounds = plan_kimi_k3_chunk_rounds(
            [127], [0], chunk_budget=64, alignment_tokens=64
        )
        plan = KimiK3BatchPlan.from_rounds(rounds)
        for item, selection in zip(rounds, plan.draft_rows):
            hidden = torch.randn(item.token_count, 4)
            selection = selection.prepare(hidden.device)
            result = selection.select(hidden)
            self.assertEqual(
                result.untyped_storage().data_ptr(), hidden.untyped_storage().data_ptr()
            )

    def test_batch_plan_does_not_retain_prepared_device_index(self):
        rounds = plan_kimi_k3_chunk_rounds(
            [4, 5], [0, 0], chunk_budget=128, alignment_tokens=64
        )
        plan = KimiK3BatchPlan.from_rounds(rounds)
        descriptor = plan.draft_rows[0]
        prepared = descriptor.prepare(torch.device("cpu"))
        self.assertIsNotNone(prepared.index)
        index_ref = weakref.ref(prepared.index)
        del prepared
        gc.collect()
        self.assertIsNone(index_ref())
        self.assertFalse(
            any(isinstance(x, torch.Tensor) for x in vars(descriptor).values())
        )
        self.assertEqual(descriptor.token_count, 7)

    def test_padding_matches_padded_batch_offsets(self):
        original = PyAttentionInputs()
        for lengths in ([5], [3, 9, 1], [64, 64], [1, 1, 1]):
            original.input_lengths_host = torch.tensor(lengths, dtype=torch.int32)
            rounds = plan_kimi_k3_chunk_rounds(
                lengths, [0] * len(lengths), chunk_budget=256, alignment_tokens=64
            )
            for round_plan in rounds:
                prepared = build_chunk_attention_inputs(
                    original, round_plan=round_plan, device=torch.device("cpu")
                )
                lens = [s.new_length for s in round_plan.slices]
                expected, offset = [], 0
                for length in lens:
                    expected.extend([offset] * length)
                    offset += max(lens) - length
                self.assertEqual(prepared.padding_offset.tolist(), expected)

    def test_plan_has_no_gpu_workspace(self):
        with patch(
            "torch.empty", side_effect=AssertionError("workspace allocated during plan")
        ):
            plan = plan_kimi_kda_prefill_metadata(
                torch.tensor([0, 65, 68]),
                torch.tensor([65, 3]),
                torch.tensor([0, 64]),
                checkpoint_tokens=64,
                local_heads=12,
                head_dim=128,
            )
        self.assertEqual(plan.required_slots, 2)
        self.assertEqual(plan.sequence_count, 2)
        self.assertEqual(plan.active_indices, (0, 1))

    def test_multimodal_mask_does_not_mutate_cache_tokens(self):
        tokens = torch.tensor([1, -10, -11, -12, 5])
        mm = SimpleNamespace(
            multimodal_features=[torch.zeros(3, 4)],
            mm_features_locs_host=torch.tensor([1]),
        )
        masked = prepare_embedding_ids(tokens, mm)
        self.assertEqual(masked.tolist(), [1, 0, 0, 0, 5])
        self.assertEqual(tokens.tolist(), [1, -10, -11, -12, 5])

    def test_draft_multimodal_shift_stays_with_its_request(self):
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor([-1, -1, 5, -1, -1, 6])
        inputs.attention_inputs = PyAttentionInputs()
        inputs.attention_inputs.cu_seqlens_host = torch.tensor(
            [0, 3, 6], dtype=torch.int32
        )
        inputs.attention_inputs.cu_seqlens = inputs.attention_inputs.cu_seqlens_host
        inputs.multimodal_inputs.multimodal_features = [
            torch.arange(8.0).reshape(2, 4),
            torch.arange(8.0, 16.0).reshape(2, 4),
        ]
        inputs.multimodal_inputs.mm_features_locs_host = torch.tensor(
            [0, 3], dtype=torch.int32
        )
        model = SimpleNamespace(hidden_size=4, embedding_dtype=torch.float32)
        prepared = prepare_draft_round(model, inputs, None)
        self.assertEqual(prepared.embedding_ids.tolist(), [0, -1, 5, 0, -1, 6])
        self.assertEqual([loc for loc, _ in prepared.embedding_injections], [0, 3])
        torch.testing.assert_close(
            prepared.embedding_injections[0][1], torch.tensor([[4.0, 5.0, 6.0, 7.0]])
        )
        torch.testing.assert_close(
            prepared.embedding_injections[1][1],
            torch.tensor([[12.0, 13.0, 14.0, 15.0]]),
        )
        self.assertEqual(
            inputs.multimodal_inputs.mm_features_locs_host.tolist(), [0, 3]
        )
        self.assertEqual(inputs.input_ids.tolist(), [-1, -1, 5, -1, -1, 6])

    def test_execution_config_is_fixed(self):
        model = SimpleNamespace(
            layer_num=4,
            layers=[],
            parallelism_config=SimpleNamespace(
                get_attn_tp_size=lambda: 8, get_attn_tp_rank=lambda: 3
            ),
        )
        with patch.dict(
            os.environ,
            {
                "SP_TYPE": "eagle3",
                "KIMI_K3_PREFILL_CHUNK_TOKENS": "128",
                "KIMI_K3_EAGLE3_AUX_LAYER_IDS": "0,1,3",
            },
        ):
            spec = KimiK3ExecutionSpec.from_model(model)
        with patch.dict(
            os.environ, {"SP_TYPE": "mtp", "KIMI_K3_PREFILL_CHUNK_TOKENS": "64"}
        ):
            self.assertEqual(spec.sp_type, "eagle3")
            self.assertEqual(spec.chunk_tokens, 128)
            self.assertEqual(spec.aux_layers, (0, 1, 3))

    def test_kv_b_reuse_is_opt_in_for_k3(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
            MlaFlashMLAPrefillOp,
        )

        backend = MlaFlashMLAPrefillOp.__new__(MlaFlashMLAPrefillOp)
        projection = object()
        backend._kv_b_projections = {3: projection}
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill.LinearFactory.create_linear_from_weights"
        ) as create:
            self.assertIs(backend._create_kv_b_proj(3), projection)
            create.assert_not_called()
            backend._kv_b_projections = None
            backend.weights = [{"weight": object()}]
            backend.quant_config = None
            self.assertIs(backend._create_kv_b_proj(0), create.return_value)
            self.assertIs(create.call_args.args[0], backend.weights[0])

    def test_mla_contexts_do_not_overwrite_module_state(self):
        from dataclasses import replace
        from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLAContext
        from rtp_llm.models_py.distributed.sequence_parallel import sequence_parallel_layout
        layout = sequence_parallel_layout(
            mode="prefill", logical_requests=1, physical_requests=1,
            tokens_per_request=0, logical_tokens=8, physical_tokens=8,
            world_size=8, rank=0,
        )
        original = KimiK3MLAContext(layout)
        projected = [torch.randn(8, 4)]
        bound = replace(original, projected_qkv_a=projected)
        self.assertIsNone(original.projected_qkv_a)
        self.assertIs(bound.projected_qkv_a, projected)
        self.assertIs(bound.sp_layout, original.sp_layout)

    def test_forward_prepared_never_replans(self):
        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        from rtp_llm.models_py.modules.kimi_k3.parallel_mode import KimiK3ParallelMode
        model.parallel_mode = KimiK3ParallelMode.TP_SP
        model.layers = nn.ModuleList([])
        model.kv_cache = None
        model._layer_group_ids = None
        model.num_attn_res_blocks = 1
        model.config = SimpleNamespace(hidden_size=4)
        model.embedding_weight = torch.empty(1, 4)
        model.execution_spec = KimiK3ExecutionSpec(1, 0, 0, "", (), frozenset(), ())
        model._embed_prepared = lambda ids, mm: ids.float().unsqueeze(1).repeat(1, 4)
        model.norm = lambda hidden, residual: hidden + 2
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor([1, 2, 3])
        inputs.attention_inputs = PyAttentionInputs()
        inputs.attention_inputs.input_lengths = torch.tensor([3], dtype=torch.int32)
        inputs.attention_inputs.is_prefill = True
        inputs.attention_inputs.cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
        inputs.attention_inputs.cu_seqlens_host = torch.tensor(
            [0, 3], dtype=torch.int32
        )
        inputs.ktp_valid_row_mask = torch.tensor([1, 0, 1], dtype=torch.int32)
        with patch(
            "rtp_llm.models_py.modules.kimi_k3.input_preparation.create_write_cache_store_impl",
            return_value=None,
        ):
            prepared = prepare_round(model, inputs)
        self.assertEqual(
            prepared.attn_meta.valid_token_mask.data_ptr(),
            inputs.ktp_valid_row_mask.data_ptr(),
        )
        inputs.ktp_valid_row_mask.fill_(1)
        torch.testing.assert_close(
            prepared.attn_meta.valid_token_mask, torch.ones(3, dtype=torch.int32)
        )
        with patch(
            "rtp_llm.models_py.model_desc.kimi_k3.prepare_round",
            side_effect=AssertionError("compute replanned"),
        ):
            result = model.forward_prepared(prepared)
        torch.testing.assert_close(
            result.hidden_states, torch.tensor([[3.0] * 4, [4.0] * 4, [5.0] * 4])
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_graph_replay_consumes_updated_device_tokens_without_reprepare(self):
        from rtp_llm.models_py.model_desc.module_base import GptModelBase

        model = KimiK3Model.__new__(KimiK3Model)
        nn.Module.__init__(model)
        from rtp_llm.models_py.modules.kimi_k3.parallel_mode import KimiK3ParallelMode
        model.parallel_mode = KimiK3ParallelMode.TP_SP
        model.layers = nn.ModuleList([])
        model.kv_cache = None
        model._layer_group_ids = None
        model.num_attn_res_blocks = 1
        model.config = SimpleNamespace(hidden_size=4)
        model.embedding_weight = torch.empty(1, 4, device="cuda")
        model.execution_spec = KimiK3ExecutionSpec(1, 0, 0, "", (), frozenset(), ())
        model._embed_prepared = (
            lambda ids, injections: ids.float().unsqueeze(1).repeat(1, 4)
        )
        model.norm = lambda hidden, residual: hidden + 2
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor([1, 2, 3], device="cuda")
        inputs.attention_inputs = PyAttentionInputs()
        inputs.attention_inputs.input_lengths = torch.tensor([3], dtype=torch.int32)
        inputs.attention_inputs.is_prefill = True
        inputs.attention_inputs.is_target_verify = True
        inputs.attention_inputs.is_cuda_graph = True
        inputs.attention_inputs.cu_seqlens = torch.tensor(
            [0, 3], dtype=torch.int32, device="cuda"
        )
        inputs.attention_inputs.cu_seqlens_host = torch.tensor(
            [0, 3], dtype=torch.int32
        )
        backend = SimpleNamespace()
        with patch.object(
            GptModelBase, "prepare_fmha_impl", return_value=backend
        ), patch(
            "rtp_llm.models_py.modules.kimi_k3.input_preparation.create_write_cache_store_impl",
            return_value=None,
        ):
            model.prepare_fmha_impl(inputs, True)
        self.assertIsNone(backend._k3_prepared_round.fmha_impl)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.forward(inputs, backend)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with patch(
            "rtp_llm.models_py.model_desc.kimi_k3.prepare_round",
            side_effect=AssertionError("replanned in graph"),
        ):
            with torch.cuda.graph(graph):
                result = model.forward(inputs, backend)[0]
            for values in ([3, 4, 5], [8, 1, 9], [0, 0, 2]):
                inputs.input_ids.copy_(torch.tensor(values, device="cuda"))
                graph.replay()
                expected = (
                    (torch.tensor(values, device="cuda").float() + 2)
                    .unsqueeze(1)
                    .repeat(1, 4)
                )
                torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
