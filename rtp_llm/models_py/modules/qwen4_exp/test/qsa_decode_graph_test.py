from types import SimpleNamespace
from unittest import TestCase, main, skipUnless

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa_runtime import Qwen4ExpQSARuntimeContext
from rtp_llm.ops import RopeStyle


@skipUnless(torch.cuda.is_available(), "CUDA is required for graph replay")
class Qwen4ExpQSADecodeGraphTest(TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.device = torch.device("cuda")
        self.lengths = torch.tensor([7], dtype=torch.int32).pin_memory()
        self.input_lengths = torch.tensor([1], dtype=torch.int32).pin_memory()
        self.prefixes = torch.empty(0, dtype=torch.int32).pin_memory()
        self.positions = torch.tensor([7, 7, 7], dtype=torch.int32, device="cuda")
        self.kv_table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
        self.state_table = self.kv_table.clone()
        self.kv = torch.zeros(3, 32 * 128 * 2, dtype=torch.uint8, device="cuda")
        self.state = torch.zeros(3, 8 * 128, dtype=torch.float32, device="cuda")
        self.state.view(3, 8, 128)[1, 4:7] = torch.randn(3, 128, device="cuda")
        self.kv.view(torch.bfloat16).view(3, 32, 128)[1, 0] = torch.randn(
            128, device="cuda"
        ).bfloat16()
        self.indexer = SimpleNamespace(
            head_dim=128,
            n_heads=4,
            compress_ratio=4,
            token_budget=8,
            k_norm_gamma=(torch.randn(128, device="cuda") * 0.1).bfloat16(),
            norm_eps=1e-6,
        )
        self.rope = SimpleNamespace(
            style=RopeStyle.Mrope,
            index_factor=3,
            dim=64,
            mrope_dim1=12,
            mrope_dim2=10,
            mrope_dim3=10,
            mrope_interleaved=True,
            base=10000,
            scale=1.0,
        )
        self.q = torch.randn(1, 4, 128, dtype=torch.bfloat16, device="cuda")
        self.raw = torch.randn(1, 128, dtype=torch.bfloat16, device="cuda")

    def _context(self, graph, kv=None, state=None, *, exact=True, draft=False):
        def inputs(kv_map=None, state_map=None):
            return SimpleNamespace(
                is_prefill=False,
                is_target_verify=False,
                is_cuda_graph=graph,
                is_exact_cuda_graph_batch=exact if graph else False,
                is_s_padded=graph,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=self.input_lengths,
                prefix_lengths=self.prefixes,
                sequence_lengths=self.lengths,
                cu_seqlens_device=None,
                cu_kv_seqlens_device=None,
                combo_position_ids=self.positions,
                kv_cache_kernel_block_id_device=kv_map,
                kv_cache_block_id_device=state_map,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=self.kv if kv is None else kv,
                seq_size_per_block=128,
            ),
            indexer_kv_inputs=inputs(kv_map=self.kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=self.state if state is None else state,
                seq_size_per_block=128,
            ),
            indexer_state_inputs=inputs(state_map=self.state_table),
            is_mtp_draft=draft,
        )

    def test_graph_requires_exact_batch_and_rejects_draft(self):
        with self.assertRaisesRegex(RuntimeError, "exact batch graph"):
            self._context(True, exact=False).select_decode_tokens(
                self.q, self.raw, indexer=self.indexer, rope_config=self.rope
            )
        with self.assertRaisesRegex(RuntimeError, "MTP draft CUDA Graph"):
            self._context(True, draft=True).select_decode_tokens(
                self.q, self.raw, indexer=self.indexer, rope_config=self.rope
            )

    def test_graph_replay_matches_eager_after_length_and_projection_update(self):
        context = self._context(True)
        context.select_decode_tokens(
            self.q, self.raw, indexer=self.indexer, rope_config=self.rope
        )
        context.finalize_side_cache()
        initial_kv = self.kv.clone()
        initial_state = self.state.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selected = context.select_decode_tokens(
                self.q, self.raw, indexer=self.indexer, rope_config=self.rope
            )
        context.finalize_side_cache()

        self.kv.copy_(initial_kv)
        self.state.copy_(initial_state)
        self.q.copy_(torch.randn_like(self.q.float()).bfloat16())
        self.raw.copy_(torch.randn_like(self.raw.float()).bfloat16())
        self.lengths[0] = 8
        self.positions.fill_(8)
        graph.replay()

        reference_kv = initial_kv.clone()
        reference_state = initial_state.clone()
        eager = self._context(False, reference_kv, reference_state)
        expected = eager.select_decode_tokens(
            self.q, self.raw, indexer=self.indexer, rope_config=self.rope
        )
        torch.testing.assert_close(selected, expected)
        torch.testing.assert_close(self.kv, reference_kv)
        torch.testing.assert_close(self.state, reference_state)


if __name__ == "__main__":
    main()
