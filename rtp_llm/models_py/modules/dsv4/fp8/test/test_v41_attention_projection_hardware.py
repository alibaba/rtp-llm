"""SM120 full-size projections and projection/cache/index/attention graph flow."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.v41_attention import V41SparseAttention
from rtp_llm.models_py.modules.dsv4.v41_attention_projection import (
    build_v41_projections,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.models_py.modules.dsv4.v41_compressor import V41PairMetadata, V41PairState
from rtp_llm.models_py.modules.dsv4.v41_indexer import V41Indexer
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import quantize_group32


def fixture(ratio, *, full=False):
    torch.manual_seed(423)
    hidden, rank, output_rank = (5120, 1280, 1024) if full else (128, 128, 32)
    device = torch.device("cuda")
    weights = {}
    for name, n, k in (
        ("attn.wq_a", rank, hidden),
        ("attn.wq_b", 32768, rank),
        ("attn.wkv", 512, hidden),
        ("attn.wo_a", output_rank * 8, 4096),
        ("attn.wo_b", hidden, output_rank * 8),
        ("indexer.wq_b", 4096, rank),
    ):
        weights[f"v4.{name}.weight"] = (torch.randn(n, k, device=device) * 2).to(
            torch.float8_e4m3fn
        )
        weights[f"v4.{name}.scale"] = torch.full(
            (n // 32, k // 32), 121, dtype=torch.uint8, device=device
        )
    for name, size in (
        ("attn.q_norm", rank),
        ("attn.kv_norm", 512),
        ("compressor.norm", 512),
    ):
        weights[f"v4.{name}.weight"] = torch.randn(size, device=device).bfloat16()
    weights["v4.attn.attn_sink"] = torch.randn(64, device=device)
    weights["v4.indexer.weights_proj.weight"] = (
        torch.randn(32, hidden, device=device) / hidden**0.5
    ).bfloat16()
    weights["v4.compressor.wkv.weight"] = (
        torch.randn(512, hidden, device=device) / hidden**0.5
    ).bfloat16()
    weights["v41.indexer.wk.weight"] = (
        torch.randn(128, 512, device=device) / 512**0.5
    ).bfloat16()
    weights["v41.indexer.k_norm.weight"] = torch.randn(128, device=device).bfloat16()
    if ratio == 2:
        weights["v4.compressor.wgate.weight"] = (
            torch.randn(512, hidden, device=device) / hidden**0.5
        ).bfloat16()
    sources = (
        V41LayerSources(2, 2, 2, 2, None)
        if ratio == 2
        else V41LayerSources(20, 1, 20, 20, 20)
    )
    frequencies = precompute_freqs_cis(
        64, 1024, 65536, 160000, 16, 32, 1, device=device
    )
    return (*build_v41_projections(sources, weights, frequencies), weights)


def norm(x, weight):
    y = x.float()
    return (
        y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-20) * weight.cpu()
    ).to(x.dtype)


def rotate(x, frequencies, inverse=False):
    freq = frequencies.cpu().conj() if inverse else frequencies.cpu()
    tail = torch.view_as_complex(x[..., -64:].float().reshape(*x.shape[:-1], 32, 2))
    result = torch.view_as_real(tail * freq[:, None]).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-64], result), -1)


def dense_reference(op, x):
    raw, scales = quantize_group32(x.cpu(), backend="eager")
    x = raw.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    # Fixture has one scale value; avoid allocating a full expanded scale matrix.
    return F.linear(x, op.weight.cpu().float() * 2**-6).bfloat16()


class V41AttentionProjectionHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))

    def test_checkpoint_geometry_and_reference_equations(self):
        op, kv, weights = fixture(1, full=True)
        x = torch.randn(2, 5120, device="cuda").bfloat16()
        pos = torch.tensor([99, 257], device="cuda")
        valid = torch.ones(2, dtype=torch.bool, device="cuda")
        frame = op.prepare(x, pos, valid)
        qr = norm(dense_reference(op.wq_a, x), op.q_norm)
        q = rotate(dense_reference(op.wq_b, qr).reshape(2, 64, 512), frame.frequencies)
        iq = rotate(
            dense_reference(op.index_wq_b, qr).reshape(2, 32, 128), frame.frequencies
        )
        torch.testing.assert_close(frame.query.cpu(), q, atol=0.06, rtol=0.03)
        torch.testing.assert_close(frame.index_query.cpu(), iq, atol=0.06, rtol=0.03)
        torch.testing.assert_close(
            frame.index_weights.cpu(),
            F.linear(x.cpu(), op.index_weights.cpu()),
            atol=0.016,
            rtol=0.016,
        )
        self.assertIs(op.wq_a.weight, weights["v4.attn.wq_a.weight"])
        torch.testing.assert_close(
            op.wo_a.flatten(0, 1),
            (weights["v4.attn.wo_a.weight"].float() * 2**-6).bfloat16(),
            atol=0,
            rtol=0,
        )
        attention = torch.randn(2, 64, 512, device="cuda").bfloat16()
        unrotated = rotate(attention.cpu(), frame.frequencies, inverse=True).reshape(
            2, 8, 4096
        )
        grouped = torch.stack(
            [F.linear(unrotated[:, g], op.wo_a[g].cpu()) for g in range(8)], 1
        ).flatten(1)
        torch.testing.assert_close(
            op.output(attention, frame).cpu(),
            dense_reference(op.wo_b, grouped),
            atol=0.125,
            rtol=0.03,
        )
        self.assertEqual(kv.prepare(x, pos, valid).global_kv.shape, (2, 512))

    def test_combined_projection_store_index_attention_output_graph(self):
        for ratio in (1, 2):
            op, kv, _ = fixture(ratio)
            device = torch.device("cuda")
            x = torch.randn(5, 128, device=device).bfloat16()
            x[-1] = float("nan")
            meta = V41PairMetadata(
                torch.tensor([0, 0, 0, 0, -999], device=device),
                torch.tensor([0, 1, 2, 3, -999], device=device),
                torch.tensor([True, True, True, True, False], device=device),
                torch.tensor([-1, 0, -1, 2, 99999], device=device),
                torch.full((5,), -1, dtype=torch.int64, device=device),
                torch.tensor([0, 1, 2, 3, 99999], device=device),
            )
            state = V41PairState(torch.zeros(2, 4096, device=device))

            def cache(tag):
                return V41PagedCache(
                    torch.zeros(
                        (2, dsv41_page_layouts(1)[tag].stride_bytes),
                        dtype=torch.uint8,
                        device=device,
                    ),
                    tag,
                )

            swa, global_kv, index_k = (
                cache("swa_kv"),
                cache("csa_kv" if ratio == 2 else "hca_kv"),
                cache("indexer_kv"),
            )
            swa_writes = torch.tensor([0, 1, 2, 3, -999], device=device)
            global_writes = torch.tensor(
                [0, 0, 1, 1, -999] if ratio == 2 else [0, 1, 2, 3, -999], device=device
            )
            index_writes = global_writes.clone()
            swa_reads = torch.full((5, 128), -1, device=device, dtype=torch.int64)
            for row in range(4):
                swa_reads[row, : row + 1] = torch.arange(row + 1, device=device)
            pages = torch.zeros(5, 1, device=device, dtype=torch.int32)
            indexer = V41Indexer(op.sources, max_positions=256 // ratio)
            attention = V41SparseAttention()

            def run():
                q = op.prepare(x, meta.positions, meta.valid_rows)
                kwargs = (
                    {"pair_state": state, "pair_metadata": meta} if ratio == 2 else {}
                )
                produced = kv.prepare(x, meta.positions, meta.valid_rows, **kwargs)
                produced.publish(
                    swa,
                    swa_writes,
                    global_cache=global_kv,
                    global_slots=global_writes,
                    index_cache=index_k,
                    index_slots=index_writes,
                )
                selected = indexer(
                    q.index_query,
                    q.index_weights,
                    index_k,
                    pages,
                    meta.positions,
                    valid_rows=meta.valid_rows,
                )
                slots = torch.where(
                    selected.topk >= 0,
                    selected.topk.long()
                    + pages[:, :1].long() * global_kv.layout.entries,
                    -1,
                )
                value = attention(
                    q.query,
                    op.sinks,
                    swa,
                    swa_reads,
                    valid_rows=meta.valid_rows,
                    global_cache=global_kv,
                    global_slots=slots,
                )
                return op.output(value, q), selected

            for _ in range(2):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual, selected = run()
            for live, page in ((True, 1), (False, 0), (True, 0)):
                x[:4].normal_()
                meta.valid_rows[:4] = live
                pages.fill_(page)
                global_writes[:4] = (
                    torch.tensor(
                        [0, 0, 1, 1] if ratio == 2 else [0, 1, 2, 3], device=device
                    )
                    + page * global_kv.layout.entries
                )
                index_writes[:4] = (
                    torch.tensor(
                        [0, 0, 1, 1] if ratio == 2 else [0, 1, 2, 3], device=device
                    )
                    + page * index_k.layout.entries
                )
                expected, expected_selection = run()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    selected.topk, expected_selection.topk, atol=0, rtol=0
                )
                self.assertTrue(bool((actual[-1] == 0).all()))
                self.assertTrue(bool(torch.isfinite(actual).all()))


if __name__ == "__main__":
    unittest.main()
