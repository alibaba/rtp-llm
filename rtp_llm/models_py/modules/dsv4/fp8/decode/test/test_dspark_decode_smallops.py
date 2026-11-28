"""CUDA parity & CUDA Graph audit for DeepSeekV4DSparkModel Track B-2 small-op optimizations."""

from __future__ import annotations

import ctypes
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc import deepseek_v4_dspark_model as dspark_mod
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    dequantize_slots_to_bf16,
)
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
from rtp_llm.test.utils.numeric_util import calc_diff


def _count_graph_nodes(graph: torch.cuda.CUDAGraph) -> dict[str, int]:
    libcudart = ctypes.CDLL("libcudart.so")
    cuda_graph_fn = libcudart.cudaGraphGetNodes
    cuda_graph_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    cuda_graph_fn.restype = ctypes.c_int

    node_type_fn = libcudart.cudaGraphNodeGetType
    node_type_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    node_type_fn.restype = ctypes.c_int

    raw = ctypes.c_void_p(graph.raw_cuda_graph())
    num = ctypes.c_size_t(0)
    assert cuda_graph_fn(raw, None, ctypes.byref(num)) == 0
    n = int(num.value)
    counts = {"total": n, "kernel": 0, "memcpy": 0, "memset": 0}
    if n == 0:
        return counts
    nodes = (ctypes.c_void_p * n)()
    assert cuda_graph_fn(raw, nodes, ctypes.byref(num)) == 0
    ntype = ctypes.c_int(-1)
    for i in range(n):
        assert node_type_fn(nodes[i], ctypes.byref(ntype)) == 0
        v = int(ntype.value)
        if v == 0:
            counts["kernel"] += 1
        elif v == 1:
            counts["memcpy"] += 1
        elif v == 2:
            counts["memset"] += 1
    return counts


class DSparkDecodeSmallOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.device = torch.device("cuda", torch.cuda.current_device())

    @torch.inference_mode()
    def test_compute_draft_hidden_states_inplace_rmsnorm_and_custom_norm_guard(
        self,
    ) -> None:
        dim = 4096
        model = dspark_mod.DeepSeekV4DSparkModel.__new__(
            dspark_mod.DeepSeekV4DSparkModel
        )
        model._v4_args = SimpleNamespace(dim=dim)
        norm_w = torch.rand(dim, dtype=torch.bfloat16, device=self.device).add_(0.5)
        real_norm = RMSNorm(norm_w, 1e-6)
        hc_proj = torch.empty(0)
        del hc_proj

        def _hc_reduce(h: torch.Tensor) -> torch.Tensor:
            # Simulates fresh non-aliased [B, G, D] output from _hc_head_reduce.
            return h.sum(dim=2).contiguous()

        model.v4 = SimpleNamespace(_hc_head_reduce=_hc_reduce, norm=real_norm)

        for bs, gamma in ((1, 4), (16, 4), (32, 4), (64, 4)):
            with self.subTest(bs=bs, gamma=gamma):
                h = torch.randn(
                    bs, gamma, 4, dim, dtype=torch.bfloat16, device=self.device
                ).mul_(0.1)
                expected = real_norm(h.sum(dim=2).reshape(bs * gamma, dim))
                got = model.compute_draft_hidden_states(h)
                self.assertEqual(tuple(got.shape), (bs * gamma, dim))
                self.assertLess(calc_diff(got.float(), expected.float()), 1.0e-5)

        # Custom instance `forward` override or subclass must bypass output= kwarg.
        class CustomNorm(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2.0

        model.v4 = SimpleNamespace(_hc_head_reduce=_hc_reduce, norm=CustomNorm())
        h_small = torch.randn(4, 4, 4, dim, dtype=torch.bfloat16, device=self.device)
        got_custom = model.compute_draft_hidden_states(h_small)
        torch.testing.assert_close(
            got_custom, h_small.sum(dim=2).reshape(16, dim) * 2.0, rtol=0.0, atol=0.0
        )

    @torch.inference_mode()
    def test_commit_layer_features_inplace_fused_rmsnorm_rope_and_cuda_graph(
        self,
    ) -> None:
        dim = 512
        head_dim = 512
        rope_dim = 64
        entries_per_block = 128
        tokens_per_block = 256
        entry_bytes = 584

        freqs = precompute_freqs_cis(
            dim=rope_dim,
            seqlen=1024,
            original_seq_len=1024,
            base=10000.0,
            factor=1.0,
            beta_fast=32,
            beta_slow=1,
            device=self.device,
        )
        kv_norm = torch.rand(head_dim, dtype=torch.bfloat16, device=self.device).add_(
            0.5
        )
        wkv_weight = torch.randn(
            dim, dim, dtype=torch.bfloat16, device=self.device
        ).mul_(0.03)
        swa_pool_got = torch.zeros(
            (16, entries_per_block, entry_bytes), dtype=torch.uint8, device=self.device
        )
        swa_pool_ref = torch.zeros_like(swa_pool_got)

        class FakeAttn:
            def __init__(self) -> None:
                self.head_dim = head_dim
                self.rope_head_dim = rope_dim
                self.eps = 1e-6
                self.kv_norm = kv_norm
                self.freqs_cis = freqs
                self.wkv = wkv_weight

            def _swa_cp_byte_sliced(self) -> bool:
                return False

            def _lin(self, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, weight.t())

        fake_attn = FakeAttn()

        def make_model(pool: torch.Tensor) -> dspark_mod.DeepSeekV4DSparkModel:
            m = dspark_mod.DeepSeekV4DSparkModel.__new__(
                dspark_mod.DeepSeekV4DSparkModel
            )

            @contextmanager
            def _bound(_layer_idx, _bt, cp_ctx=None):
                yield fake_attn, entries_per_block, pool

            m._swa_cache_bound = _bound  # type: ignore[method-assign]
            return m

        m_got = make_model(swa_pool_got)

        bs = 16
        gamma_plus_1 = 4
        n_tok = bs * gamma_plus_1
        main_x = torch.randn(n_tok, dim, dtype=torch.bfloat16, device=self.device).mul_(
            0.2
        )
        ctx_req_ids = (
            torch.arange(bs, dtype=torch.int32, device=self.device)
            .unsqueeze(1)
            .expand(bs, gamma_plus_1)
            .reshape(-1)
            .contiguous()
        )
        ctx_pos = (
            (
                (
                    torch.arange(bs, dtype=torch.int32, device=self.device) * 7 + 10
                ).unsqueeze(1)
                + torch.arange(
                    gamma_plus_1, dtype=torch.int32, device=self.device
                ).unsqueeze(0)
            )
            .reshape(-1)
            .contiguous()
        )
        committed_ends = (
            torch.arange(bs, dtype=torch.int32, device=self.device) * 7 + 14
        )
        block_table = torch.arange(
            1, bs + 1, dtype=torch.int32, device=self.device
        ).view(bs, 1)

        # Reference non-inplace fused_rmsnorm_rope + write
        def run_ref() -> torch.Tensor:
            kv_proj = fake_attn._lin(fake_attn.wkv, main_x)
            ctx_kv = fused_rmsnorm_rope(
                kv_proj,
                fake_attn.kv_norm,
                fake_attn.freqs_cis,
                rope_dim,
                eps=1e-6,
                position_ids=ctx_pos,
                inplace=False,
            )
            slots = torch.arange(n_tok, dtype=torch.long, device=self.device)
            dspark_mod.decode_write_swa_fp8(
                kv=ctx_kv,
                slot_mapping=slots,
                swa_pool_3d=swa_pool_ref,
                bsz=n_tok,
                q_len=1,
                head_dim=head_dim,
            )
            return slots

        ref_slots = run_ref()
        m_got._commit_layer_features(
            layer_idx=0,
            main_x=main_x,
            context_req_ids=ctx_req_ids,
            context_positions=ctx_pos,
            committed_ends=committed_ends,
            block_table=block_table,
            tokens_per_block=tokens_per_block,
            batch_size=bs,
            precomputed_slots=(entries_per_block, ref_slots),
        )
        got_vals = dequantize_slots_to_bf16(swa_pool_got, ref_slots)
        ref_vals = dequantize_slots_to_bf16(swa_pool_ref, ref_slots)
        self.assertLess(calc_diff(got_vals.float(), ref_vals.float()), 1.0e-5)

        # CUDA graph capture + replay audit (zero memset/memcpy)
        for _ in range(3):
            m_got._commit_layer_features(
                layer_idx=0,
                main_x=main_x,
                context_req_ids=ctx_req_ids,
                context_positions=ctx_pos,
                committed_ends=committed_ends,
                block_table=block_table,
                tokens_per_block=tokens_per_block,
                batch_size=bs,
                precomputed_slots=(entries_per_block, ref_slots),
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            m_got._commit_layer_features(
                layer_idx=0,
                main_x=main_x,
                context_req_ids=ctx_req_ids,
                context_positions=ctx_pos,
                committed_ends=committed_ends,
                block_table=block_table,
                tokens_per_block=tokens_per_block,
                batch_size=bs,
                precomputed_slots=(entries_per_block, ref_slots),
            )
        torch.cuda.synchronize()
        nodes = _count_graph_nodes(graph)
        print(f"[AUDIT dspark_commit_layer bs={bs}] nodes={nodes}")
        self.assertEqual(nodes["memset"], 0)
        self.assertEqual(nodes["memcpy"], 0)

        main_x.copy_(torch.randn_like(main_x).mul_(0.3))
        run_ref()
        graph.replay()
        torch.cuda.synchronize()
        got_vals2 = dequantize_slots_to_bf16(swa_pool_got, ref_slots)
        ref_vals2 = dequantize_slots_to_bf16(swa_pool_ref, ref_slots)
        self.assertLess(calc_diff(got_vals2.float(), ref_vals2.float()), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
