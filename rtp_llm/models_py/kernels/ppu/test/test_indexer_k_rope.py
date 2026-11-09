"""K-only producer parity and the cached-prefix short CP boundary."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.kernels.ppu.indexer_k_quant import quantize_and_cache_indexer_k
from rtp_llm.models_py.kernels.ppu.indexer_k_rope import fused_indexer_k_rope
from rtp_llm.models_py.modules import LayerNorm
from rtp_llm.models_py.modules.base.cuda import indexer_op
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.hybrid.indexer import Indexer
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
    fused_qk_rope_quant,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops


class IndexerKeyOnlyTest(TestCase):
    def setUp(self):
        torch.manual_seed(20492048)
        angle = torch.randn(4096, 32, device="cuda")
        self.cos_sin = torch.cat((angle.cos(), angle.sin()), dim=-1)

    def test_key_and_cache_bytes_match_fused_qk(self):
        for rows in (0, 1, 17, 128, 256):
            for neox in (False, True):
                with self.subTest(rows=rows, neox=neox):
                    key = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)[
                        :, :128
                    ]
                    positions = (
                        torch.arange(rows, device="cuda", dtype=torch.int64) * 13 % 4096
                    )
                    query = torch.randn(
                        rows, 32, 128, device="cuda", dtype=torch.bfloat16
                    )
                    expected = fused_qk_rope_quant(
                        query,
                        key,
                        positions,
                        self.cos_sin,
                        32,
                        128,
                        64,
                        is_neox_style=neox,
                        scale_fmt="ue8m0",
                    )[2]
                    actual = fused_indexer_k_rope(key, positions, self.cos_sin, neox)
                    self.assertTrue(torch.equal(actual, expected))
                    slots = torch.arange(rows, device="cuda", dtype=torch.int64) * 2
                    cache = torch.full(
                        ((2 * rows + 63) // 64 + 1, 64, 132),
                        173,
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    reference, candidate = cache.clone(), cache.clone()
                    quantize_and_cache_indexer_k(expected, reference, slots)
                    quantize_and_cache_indexer_k(actual, candidate, slots)
                    self.assertTrue(torch.equal(candidate, reference))
                    self.assertTrue(torch.equal(candidate[-1], cache[-1]))

    def test_graph_reads_new_keys_and_positions_on_nondefault_stream(self):
        key = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(17, device="cuda", dtype=torch.int64)
        for neox in (False, True):
            for _ in range(3):
                fused_indexer_k_rope(key, positions, self.cos_sin, neox)
            torch.cuda.synchronize()
            graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            with torch.cuda.graph(graph, stream=stream):
                output = fused_indexer_k_rope(key, positions, self.cos_sin, neox)
            for _ in range(3):
                key.normal_()
                positions.copy_((positions + 113) % 4096)
                torch.cuda.synchronize()
                graph.replay()
                expected = fused_qk_rope_quant(
                    torch.ones(17, 1, 128, device="cuda", dtype=torch.bfloat16),
                    key,
                    positions,
                    self.cos_sin,
                    1,
                    128,
                    64,
                    is_neox_style=neox,
                )[2]
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(output, expected))

    def _module(self):
        module = Indexer.__new__(Indexer)
        nn.Module.__init__(module)
        module.parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: True)
        )
        module._fuse_kernels_enabled = True
        module._fuse_logits_head_gate = False
        module.index_n_heads, module.index_head_dim, module.index_topk = 32, 128, 2048
        module.softmax_scale, module.weights_scale = 128**-0.5, 32**-0.5
        module._kv_owner_tokens_per_block = 64
        module.cos_sin_cache, module.is_neox_style = self.cos_sin, False
        module.wq_b = nn.Linear(
            16, 32 * 128, bias=False, device="cuda", dtype=torch.bfloat16
        )
        module.wk = nn.Linear(32, 128, bias=False, device="cuda", dtype=torch.bfloat16)
        module.k_norm = LayerNorm(
            torch.ones(128, device="cuda", dtype=torch.bfloat16),
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
            eps=1e-6,
        )
        module.weights_proj = nn.Linear(
            32, 32, bias=False, device="cuda", dtype=torch.float32
        )
        module.weights_proj.weight.data.abs_()
        module.indexer_op = IndexerOp(
            32,
            128,
            2048,
            64,
            cos_sin_cache=self.cos_sin,
            scale_fmt="none",
            query_scale_fmt="ue8m0",
            is_neox_style=False,
            stable_topk=True,
            enable_fuse_kernels=True,
        )
        return module

    def test_native_cp_metadata_prefix_boundary_and_permuted_queries(self):
        # Use the real native prefix/slot metadata and real FP8 K writer. Only
        # all-gather is local here; the distributed fixture covers eight ranks.
        for prefix in (2032, 2033):
            with self.subTest(prefix=prefix), torch.inference_mode():
                module = self._module()
                pages = (prefix + 16 + 63) // 64
                inputs = PyAttentionInputs()
                inputs.is_prefill = True
                inputs.input_lengths = torch.tensor([16, 16], dtype=torch.int32)
                inputs.prefix_lengths = torch.tensor([prefix, 8], dtype=torch.int32)
                inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
                inputs.context_total_kv_length = prefix + 16 + 24
                inputs.kv_cache_kernel_block_id = torch.tensor(
                    [list(range(pages, 0, -1)), [0] * pages], dtype=torch.int32
                )
                inputs.kv_cache_kernel_block_id_device = (
                    inputs.kv_cache_kernel_block_id.cuda()
                )
                inputs.cu_kv_seqlens_device = torch.tensor(
                    [0, prefix + 16, inputs.context_total_kv_length],
                    dtype=torch.int32,
                    device="cuda",
                )
                params = rtp_llm_ops.SparseMlaParams()
                params.fill_params(inputs, 64)
                self.assertEqual(params.prefill_max_kv_len, prefix + 16)
                local_ids = torch.arange(32, device="cuda").roll(7)
                positions = torch.empty(32, device="cuda", dtype=torch.int64)
                positions[local_ids] = torch.cat(
                    (
                        torch.arange(prefix, prefix + 16, device="cuda"),
                        torch.arange(8, 24, device="cuda"),
                    )
                )
                cp = SimpleNamespace(
                    full_rope_pos_ids=positions,
                    kv_restore_unpad_indices=local_ids,
                    total_local_ids=local_ids,
                    total_kv_len=inputs.context_total_kv_length,
                    cu_kv_seqlens_global=inputs.cu_kv_seqlens_device,
                    precomputed_ks=params.ks,
                    precomputed_ke=params.ke,
                    precomputed_lengths=params.expanded_seq_lens,
                    precomputed_topk_off=params.topk_indices_offset,
                    kv_cache_sharded=False,
                    cp_size=1,
                    cp_rank=0,
                )
                cache_bytes = torch.zeros(
                    pages + 1, 64 * 132, device="cuda", dtype=torch.uint8
                )
                old_cache, new_cache = [
                    LayerKVCache(cache_bytes.clone(), 64) for _ in range(2)
                ]
                x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16).abs_()
                q_lora = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
                with patch.object(
                    indexer_op, "all_gather", side_effect=lambda key, **_k: key
                ):
                    q_fp8, scale = module._fused_forward_prefill_cp(
                        q_lora, x, old_cache, params, cp
                    )
                    weights = module._get_logits_head_gate(x, scale)
                    expected = module._compute_topk(
                        q_fp8, weights, old_cache, params, inputs, cp
                    )
                    with patch.object(
                        module.wq_b, "forward", wraps=module.wq_b.forward
                    ) as query, patch.object(
                        module.weights_proj,
                        "forward",
                        wraps=module.weights_proj.forward,
                    ) as head_gate:
                        actual = module(x, q_lora, new_cache, params, inputs, False, cp)
                    self.assertEqual(query.call_count, int(prefix == 2033))
                    self.assertEqual(head_gate.call_count, int(prefix == 2033))
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(
                    torch.equal(new_cache.kv_cache_base, old_cache.kv_cache_base)
                )
                self.assertFalse(torch.equal(new_cache.kv_cache_base, cache_bytes))
                if prefix == 2033:
                    self.assertTrue(bool((expected[15] >= 2048).any()))

    def test_shared_cpu_bound_includes_prefix_and_requires_ppu_stable_selection(self):
        operator = self._module().indexer_op
        for maximum, fit in ((-1, False), (0, True), (2048, True), (2049, False)):
            self.assertEqual(
                operator._prefill_all_keys_fit(
                    SimpleNamespace(prefill_max_kv_len=maximum), 32
                ),
                fit,
            )
        self.assertFalse(operator._prefill_all_keys_fit(SimpleNamespace(), 2049))
        operator._ppu_stable_topk = False
        self.assertFalse(
            operator._prefill_all_keys_fit(SimpleNamespace(prefill_max_kv_len=1024), 32)
        )


if __name__ == "__main__":
    main()
