"""FP32 head gates must survive the complete indexer score/selection path."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import quantize_indexer_k
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import PoolBackedModule
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8


class _CachedKeys:
    """The fixture provides already compressed keys in the real packed pool."""

    kpool_mode = True
    freqs_cis = None

    def set_pool_context(self, *args, **kwargs):
        pass

    def clear_pool_context(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def forward_decode_vectorized(self, *args, **kwargs):
        pass

    def finish_prefill(self, pending):
        pass


def fixture(q_len=1, dtype=torch.float32):
    model = IndexerFP8.__new__(IndexerFP8)
    PoolBackedModule.__init__(model)
    model.compress_ratio, model.index_topk = 4, 1024
    model.n_heads, model.head_dim, model.rope_head_dim = 32, 128, 0
    model.rotate_q, model.freqs_cis = True, None
    model.prefill_topk_backend = model.decode_topk_backend = "topk_v3_tie_break"
    model._glm_weights_chunk_rows = 0
    model.compressor = _CachedKeys()
    model.wq_b = torch.nn.Linear(
        2, 32 * 128, bias=False, device="cuda", dtype=torch.bfloat16
    )
    model.wq_b.requires_grad_(False)
    model.wq_b.weight.zero_()
    model.wq_b.weight[0, 0] = 1
    model.wq_b.weight[128 + 1, 0] = 1
    # Both gates round to1 in BF16, despite their strict FP32 ordering.
    model.weights_proj = torch.zeros(32, 2, device="cuda", dtype=dtype)
    model.weights_proj[0, 0], model.weights_proj[1, 0] = 1.001953125, 1.0029296875
    width, block = 1026, 64
    slots = 17 * block
    key = torch.zeros(slots, 128, device="cuda")
    first = torch.ones(128, device="cuda")
    second = first.clone()
    second[1::2] = -1
    key[:1023] = first + second  # Strictly above either boundary candidate.
    key[1023], key[1024] = first, second
    pool = torch.zeros(17, block, 132, device="cuda", dtype=torch.uint8)
    quantize_indexer_k(
        key.to(torch.bfloat16),
        torch.arange(slots, device="cuda", dtype=torch.int64),
        pool,
    )
    # Physical blocks group all K bytes before all per-token scales; the
    # [blocks, block,132] view does not imply interleaved K/scale records.
    packed = pool.view(17, block * 132)
    key = packed[:, : block * 128].reshape(slots, 128).view(torch.float8_e4m3fn)
    key_scale = packed[:, block * 128 :].view(torch.float32).reshape(slots)
    model._kv_pool_view = pool
    model._kv_block_table = torch.arange(17, device="cuda", dtype=torch.int32).view(
        1, 17
    )
    model._kv_eb = block
    model._kv_tokens_per_block = model._kv_owner_tokens_per_block = block * 4

    def gather(meta, quantized, scales):
        quantized.copy_(key[:width])
        scales.view(torch.float32).reshape(-1).copy_(key_scale[:width])

    model._gather_prefill_k_cache = gather
    x = torch.zeros(q_len, 2, device="cuda", dtype=torch.bfloat16)
    x[:, 0] = 1
    meta = SimpleNamespace(
        M=q_len,
        T=width,
        sp_int=0,
        freqs_cis_slice=None,
        compressor_meta=None,
        ks=torch.zeros(q_len, device="cuda", dtype=torch.int32),
        ke=torch.full((q_len,), width, device="cuda", dtype=torch.int32),
    )
    return model, x, meta


def reference(rows, higher_head=1):
    # Hadamard head0 is constant, head1 alternates signs. The first1023
    # keys score as the sum of the two positive gates; candidateA uses only
    # head0 and candidateB only head1. Therefore FP32 must choose B, index1024.
    expected = torch.arange(1024, dtype=torch.int32).expand(rows, -1).clone()
    if higher_head == 1:
        expected[:, -1] = 1024
    return expected


def decode(model, x, q_len=1):
    width = 1026
    start = torch.tensor([width * 4 - q_len], device="cuda", dtype=torch.int32)
    positions = start[:, None] + torch.arange(q_len, device="cuda", dtype=torch.int32)
    meta = SimpleNamespace(
        positions=positions, compressed_lens_per_token=(positions + 1) // 4
    )
    out = torch.empty(1, q_len, 1024, device="cuda", dtype=torch.int32)
    model.forward_decode_vectorized(
        x.view(1, q_len, 2),
        x.view(1, q_len, 2),
        start,
        out,
        position_ids=positions.reshape(-1) if q_len > 1 else None,
        compressor_meta=meta if q_len > 1 else None,
    )
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53IndexerGatePrecisionTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {"DSV4_INDEXER_TOPK_CANONICALIZE": "1", "DSV4_PERSISTENT_TOPK": "1"},
        )
        env.start()
        self.addCleanup(env.stop)

    def test_fp32_prefill_gate_selects_the_strictly_higher_boundary_key(self):
        for fused in ("0", "1"):
            for pending in (False, True):
                with self.subTest(fused=fused, pending=pending), patch.dict(
                    os.environ, {"GLM53_INDEXER_FUSED_Q_QUANT": fused}
                ):
                    model, x, meta = fixture()
                    if pending:
                        actual = model.forward_with_pending_nested(x, x, meta, None)
                    else:
                        actual = model.forward(x, x, meta, workspace=None)
                    torch.testing.assert_close(
                        actual.cpu(), reference(1), rtol=0, atol=0
                    )

    def test_fp32_decode_and_mtp_gate_select_the_strictly_higher_boundary_key(self):
        for fused in ("0", "1"):
            for q_len in (1, 4):
                with self.subTest(fused=fused, q_len=q_len), patch.dict(
                    os.environ, {"GLM53_INDEXER_FUSED_Q_QUANT": fused}
                ):
                    model, x, _ = fixture(q_len)
                    actual = decode(model, x, q_len)
                    torch.testing.assert_close(
                        actual.reshape(q_len, -1).cpu(),
                        reference(q_len),
                        rtol=0,
                        atol=0,
                    )

    def test_bf16_gates_retain_the_existing_tie_rule(self):
        for fused in ("0", "1"):
            with self.subTest(fused=fused), patch.dict(
                os.environ, {"GLM53_INDEXER_FUSED_Q_QUANT": fused}
            ):
                model, x, meta = fixture(dtype=torch.bfloat16)
                actual = model.forward(x, x, meta, workspace=None)
                torch.testing.assert_close(
                    actual.cpu(), reference(1, higher_head=0), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    decode(model, x).reshape(1, -1).cpu(),
                    reference(1, higher_head=0),
                    rtol=0,
                    atol=0,
                )

    def test_decode_graph_replays_changed_fp32_gate_order(self):
        for fused in ("0", "1"):
            with self.subTest(fused=fused), patch.dict(
                os.environ, {"GLM53_INDEXER_FUSED_Q_QUANT": fused}
            ):
                model, x, _ = fixture()
                start = torch.tensor([1026 * 4 - 1], device="cuda", dtype=torch.int32)
                actual = torch.empty(1, 1, 1024, device="cuda", dtype=torch.int32)

                def run():
                    model.forward_decode_vectorized(
                        x.view(1, 1, 2), x.view(1, 1, 2), start, actual
                    )

                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for higher in (1, 0, 1):
                    values = [1.001953125, 1.001953125]
                    values[higher] = 1.0029296875
                    model.weights_proj[:2, 0].copy_(torch.tensor(values, device="cuda"))
                    graph.replay()
                    torch.testing.assert_close(
                        actual.reshape(1, -1).cpu(),
                        reference(1, higher_head=higher),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
