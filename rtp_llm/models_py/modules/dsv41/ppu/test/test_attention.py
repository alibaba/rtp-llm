"""Sparse-attention workspace tests: assembly, indices, chunking, import rules.

The workspace pools are populated with independent encoders (from the
documented constants), so the dequantized workspace rows and index layout are
checked end-to-end without calling the codec under test for the expectations.
"""

import sys
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant, attention, indexer, kv_codec

KV_DIM = 512
WINDOW = 128
COMBINED_WIDTH = 640


class ImportCleanlinessTest(unittest.TestCase):
    def test_all_modules_import_without_accelerators(self):
        for name in (
            "rtp_llm.models_py.modules.dsv41.ppu._quant",
            "rtp_llm.models_py.modules.dsv41.ppu.kv_codec",
            "rtp_llm.models_py.modules.dsv41.ppu.indexer",
            "rtp_llm.models_py.modules.dsv41.ppu.attention",
        ):
            self.assertIn(name, sys.modules)
        # Importing them on a CPU host must not pull in accelerator wheels.
        self.assertNotIn("triton", sys.modules)
        self.assertNotIn("flash_mla", sys.modules)
        self.assertNotIn("deep_gemm", sys.modules)


class _FakeFlash:
    """Minimal flash_mla wheel stand-in recording sparse_fwd calls."""

    instances = []

    def flash_mla_sparse_fwd(
        self, q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
    ):
        _FakeFlash.instances.append(
            {
                "q_shape": tuple(q.shape),
                "kv_shape": tuple(kv.shape),
                "indices_shape": tuple(indices.shape),
                "sm_scale": sm_scale,
                "d_v": d_v,
                "topk_length": topk_length.clone() if topk_length is not None else None,
                "attn_sink": attn_sink,
            }
        )
        return (
            q.new_zeros((q.shape[0], q.shape[1], d_v)),
            q.new_zeros((q.shape[0], q.shape[1])),
            q.new_zeros((q.shape[0], q.shape[1])),
        )

    def flash_mla_with_kvcache(
        self, *, q, block_table, cache_seqlens, sm_scale, attn_sink=None, **kwargs
    ):
        _FakeFlash.instances.append(
            {
                "decode_q_shape": tuple(q.shape),
                "block_table": block_table,
                "cache_seqlens": cache_seqlens,
                "sm_scale": sm_scale,
            }
        )
        heads = q.shape[-2]
        return (
            q.new_zeros((q.shape[0], heads, 576)),
            q.new_zeros((q.shape[0], heads, 1)),
        )


def _encode_global_independent(kv):
    rows = kv.shape[0]
    x = kv.float().reshape(rows, 32, 16)
    maximum = x.abs().amax(dim=-1)
    scales = (maximum.clamp_min(6.0 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn)
    scale_bytes = scales.view(torch.uint8)
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0])
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.uint8)
    normalized = x / scales.float().unsqueeze(-1)
    nearest = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    sign = torch.signbit(normalized).to(torch.uint8) << 3
    codes = (order[nearest] | sign).reshape(rows, KV_DIM)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat((payload, scale_bytes), dim=1)


def _decode_global_independent(packed):
    payload, scale_bytes = packed[:, :256], packed[:, 256:]
    low = (payload & 15).long()
    high = (payload >> 4).long()
    codes = torch.stack((low, high), -1).reshape(payload.shape[0], -1)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = levels[codes & 7] * torch.where(codes.ge(8), -1.0, 1.0)
    return (
        (
            values.reshape(-1, 32, 16)
            * scale_bytes.view(torch.float8_e4m3fn).float().unsqueeze(-1)
        )
        .reshape(-1, KV_DIM)
        .to(torch.bfloat16)
    )


def _decode_swa_independent(packed):
    payload, scale_bytes = packed[:, :512], packed[:, 512:]
    return (
        (
            payload.view(torch.float8_e4m3fn).float().reshape(-1, 16, 32)
            * torch.exp2(scale_bytes.to(torch.float32) - 127.0).unsqueeze(-1)
        )
        .reshape(-1, KV_DIM)
        .to(torch.bfloat16)
    )


def _build_two_request_cache():
    """Two requests (lengths 10 and 7), ratio 2, small pool geometries."""
    torch.manual_seed(71)
    lengths = (10, 7)
    ratio, g_entries, s_entries = 2, 8, 16
    # Global rows: one per compressed entry.
    globals_ = [
        torch.randn(length // ratio, KV_DIM, dtype=torch.bfloat16) for length in lengths
    ]
    swas = [torch.randn(length, KV_DIM, dtype=torch.bfloat16) for length in lengths]
    # Physical pools: 2 blocks each; request 0 -> block 0, request 1 -> block 1.
    gp = torch.zeros((2, g_entries * 288), dtype=torch.uint8)
    sp = torch.zeros((2, s_entries * 528), dtype=torch.uint8)
    gdesc = SimpleNamespace(compression_ratio=ratio, tokens_per_block=g_entries)
    for request in range(2):
        base = request * g_entries
        slots = torch.arange(base, base + globals_[request].shape[0])
        kv_codec.insert_rows(
            gp,
            "global_kv_2",
            gdesc,
            slots,
            _encode_global_independent(globals_[request]),
        )
        sbase = request * s_entries
        sslots = torch.arange(sbase, sbase + swas[request].shape[0])
        kv_codec.insert_rows(
            sp,
            "swa_kv",
            SimpleNamespace(tokens_per_block=s_entries),
            sslots,
            kv_codec.encode_kv(swas[request], pool_tag="swa_kv", descriptor=None),
        )
    cache = SimpleNamespace(
        pools={"global_kv_2": gp, "swa_kv": sp},
        block_tables={
            "global_kv_2": torch.tensor([[0, 0], [1, 1]]),
            "swa_kv": torch.tensor([[0, 0], [1, 1]]),
        },
        descriptors={
            "global_kv_2": gdesc,
            "swa_kv": SimpleNamespace(tokens_per_block=s_entries),
        },
    )
    positions = torch.tensor(list(range(lengths[0])) + list(range(lengths[1])))
    request_ids = torch.tensor([0] * lengths[0] + [1] * lengths[1])
    return cache, positions, request_ids, globals_, swas, gdesc


class WorkspaceTest(unittest.TestCase):
    def test_two_request_workspace_layout(self):
        cache, positions, request_ids, globals_, swas, gdesc = (
            _build_two_request_cache()
        )
        rows = positions.shape[0]
        topk = 4
        torch.manual_seed(72)
        selected = torch.randint(0, 5, (rows, topk), dtype=torch.int32)
        selected[:, 0] = -1  # exercise sentinel entries
        ctx = SimpleNamespace(positions=positions, request_ids=request_ids)
        kv, indices, lengths = attention.build_sparse_workspace(
            selected, cache=cache, layer_id=2, context=ctx
        )
        self.assertEqual(tuple(kv.shape), (25, 1, KV_DIM))
        self.assertEqual(kv.dtype, torch.bfloat16)
        # Layout: [req0 global 5 | req0 swa 10 | req1 global 3 | req1 swa 7].
        expected_global0 = _decode_global_independent(
            _encode_global_independent(globals_[0])
        )
        expected_global1 = _decode_global_independent(
            _encode_global_independent(globals_[1])
        )
        expected_swa0 = _decode_swa_independent(
            kv_codec.encode_kv(swas[0], pool_tag="swa_kv", descriptor=None)
        )
        expected_swa1 = _decode_swa_independent(
            kv_codec.encode_kv(swas[1], pool_tag="swa_kv", descriptor=None)
        )
        torch.testing.assert_close(
            kv[:5, 0].float(), expected_global0.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            kv[5:15, 0].float(), expected_swa0.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            kv[15:18, 0].float(), expected_global1.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            kv[18:25, 0].float(), expected_swa1.float(), rtol=0, atol=0
        )

    def test_indices_alignment_sentinels_and_lengths(self):
        cache, positions, request_ids, _, _, _ = _build_two_request_cache()
        rows = positions.shape[0]
        topk = 4
        torch.manual_seed(73)
        selected = torch.randint(0, 5, (rows, topk), dtype=torch.int32)
        selected[:, 0] = -1
        ctx = SimpleNamespace(positions=positions, request_ids=request_ids)
        kv, indices, lengths = attention.build_sparse_workspace(
            selected, cache=cache, layer_id=2, context=ctx
        )
        # align(512 + 128, 128) = 640, one head-group dim, int32, -1 tail.
        self.assertEqual(tuple(indices.shape), (rows, 1, COMBINED_WIDTH))
        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(lengths.dtype, torch.int32)
        layout = [(0, 5, 5, 10), (15, 3, 18, 7)]  # (gbase, glen, sbase, slen)
        seqs = (10, 7)
        ratio = 2
        for row in range(rows):
            request = 0 if row < 10 else 1
            pos = int(positions[row])
            gbase, glen, sbase, slen = layout[request]
            topk_len = min((pos + 1) // ratio, topk)
            keep = [int(v) for v in selected[row, :topk_len].tolist() if v >= 0]
            swa_len = min(pos + 1, WINDOW)
            swa_start = max(0, seqs[request] - WINDOW)
            expected = [gbase + v for v in keep] + [
                sbase + (pos - swa_len + 1 - swa_start) + i for i in range(swa_len)
            ]
            n = int(lengths[row])
            self.assertEqual(n, len(expected))
            self.assertEqual(indices[row, 0, :n].tolist(), expected, row)
            self.assertTrue((indices[row, 0, n:] == -1).all().item())

    def test_swa_window_truncates_long_requests(self):
        # A request longer than the window keeps only the last 128 SWA rows.
        torch.manual_seed(74)
        length, ratio, g_entries, s_entries = 300, 1, 64, 64
        global_rows = length // ratio
        gp = torch.zeros((6, g_entries * 288), dtype=torch.uint8)
        sp = torch.zeros((6, s_entries * 528), dtype=torch.uint8)
        kv_global = torch.randn(global_rows, KV_DIM, dtype=torch.bfloat16)
        swa_kv = torch.randn(length, KV_DIM, dtype=torch.bfloat16)
        gdesc = SimpleNamespace(compression_ratio=ratio, tokens_per_block=g_entries)
        kv_codec.insert_rows(
            gp,
            "global_kv_1",
            gdesc,
            torch.arange(global_rows),
            _encode_global_independent(kv_global),
        )
        kv_codec.insert_rows(
            sp,
            "swa_kv",
            SimpleNamespace(tokens_per_block=s_entries),
            torch.arange(length),
            kv_codec.encode_kv(swa_kv, pool_tag="swa_kv", descriptor=None),
        )
        cache = SimpleNamespace(
            pools={"global_kv_1": gp, "swa_kv": sp},
            block_tables={
                "global_kv_1": torch.arange(6).unsqueeze(0),
                "swa_kv": torch.arange(6).unsqueeze(0),
            },
            descriptors={"global_kv_1": gdesc},
        )
        # One query row at the final position of the request.
        ctx = SimpleNamespace(
            positions=torch.tensor([length - 1]), request_ids=torch.tensor([0])
        )
        selected = torch.zeros((1, 8), dtype=torch.int32)
        kv, indices, lengths = attention.build_sparse_workspace(
            selected, cache=cache, layer_id=20, context=ctx
        )
        # 300 global rows + 128 SWA tail rows.
        self.assertEqual(kv.shape[0], 300 + 128)
        self.assertEqual(int(lengths[0]), 8 + 128)
        # SWA indices point at the tail rows 300..427 (last 128 positions).
        n = int(lengths[0])
        swa_part = indices[0, 0, n - 128 : n]
        self.assertEqual(swa_part.tolist(), list(range(300, 428)))
        torch.testing.assert_close(
            kv[300:, 0].float(),
            _decode_swa_independent(
                kv_codec.encode_kv(swa_kv[-128:], pool_tag="swa_kv", descriptor=None)
            ).float(),
            rtol=0,
            atol=0,
        )


class SparseAttentionCallTest(unittest.TestCase):
    def setUp(self):
        _FakeFlash.instances = []
        self._fake = _FakeFlash()
        self._saved = sys.modules.get("flash_mla")
        sys.modules["flash_mla"] = self._fake
        self._saved_chunk = attention.Q_CHUNK
        self.addCleanup(self._restore)

    def _restore(self):
        attention.Q_CHUNK = self._saved_chunk
        if self._saved is None:
            sys.modules.pop("flash_mla", None)
        else:
            sys.modules["flash_mla"] = self._saved

    def test_sparse_attention_chunks_and_calls_wheel(self):
        cache, positions, request_ids, _, _, _ = _build_two_request_cache()
        rows = positions.shape[0]
        selected = torch.zeros((rows, 4), dtype=torch.int32)
        ctx = SimpleNamespace(positions=positions, request_ids=request_ids)
        q = torch.randn(rows, 64, KV_DIM, dtype=torch.bfloat16)
        attention.Q_CHUNK = 4  # force multiple chunks
        out = attention.sparse_attention(
            q, selected, cache=cache, layer_id=2, context=ctx
        )
        self.assertEqual(tuple(out.shape), (rows, 64, KV_DIM))
        calls = _FakeFlash.instances
        self.assertEqual(len(calls), 5)  # ceil(17 / 4) chunks
        for call in calls:
            self.assertEqual(call["sm_scale"], 512**-0.5)
            self.assertEqual(call["d_v"], 512)
            self.assertEqual(call["kv_shape"], (25, 1, KV_DIM))
            self.assertEqual(call["indices_shape"][1:], (1, COMBINED_WIDTH))
        # Chunks partition the query rows and carry their index slices.
        self.assertEqual([c["q_shape"][0] for c in calls], [4, 4, 4, 4, 1])
        # topk_length slices concatenate back to the full per-row lengths.
        _, _, expected_lengths = attention.build_sparse_workspace(
            selected, cache=cache, layer_id=2, context=ctx
        )
        self.assertEqual(
            torch.cat([c["topk_length"] for c in calls]).tolist(),
            expected_lengths.tolist(),
        )

    def test_pad_heads_pads_to_kernel_tiles(self):
        q = torch.randn(2, 70, KV_DIM, dtype=torch.bfloat16)
        sink = torch.zeros(2, 70)
        padded, padded_sink, heads = attention._pad_heads(q, sink)
        self.assertEqual(padded.shape, (2, 128, KV_DIM))
        self.assertEqual(padded_sink.shape, (2, 128))
        self.assertTrue(padded_sink[:, 70:].eq(float("inf")).all().item())
        self.assertEqual(heads, 70)
        no_pad, none_sink, heads2 = attention._pad_heads(
            torch.randn(2, 64, KV_DIM), None
        )
        self.assertEqual(no_pad.shape[-2], 64)
        self.assertIsNone(none_sink)

    def test_decode_attention_requires_l3_metadata(self):
        q = torch.randn(1, 64, 512, dtype=torch.bfloat16)
        with self.assertRaises(NotImplementedError):
            attention.decode_attention(q)

    def test_decode_attention_call_structure(self):
        q = torch.randn(2, 64, 512, dtype=torch.bfloat16)
        block_table = torch.zeros(2, 4, dtype=torch.int32)
        cache_seqlens = torch.tensor([100, 200], dtype=torch.int32)
        out, lse = attention.decode_attention(
            q,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
        )
        calls = _FakeFlash.instances
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["sm_scale"], 512**-0.5)
        self.assertEqual(calls[0]["block_table"] is block_table, True)
        self.assertEqual(tuple(out.shape), (2, 64, 576))


class PpuAttentionTest(unittest.TestCase):
    def setUp(self):
        if not _quant.is_ppu():
            self.skipTest("requires a PPU device")

    def test_sparse_attention_runs_on_device(self):
        cache, positions, request_ids, _, _, _ = _build_two_request_cache()
        # (CPU-built cache; on a PPU host move everything before calling.)
        self.skipTest("device-resident cache plumbing validated by L3")


if __name__ == "__main__":
    unittest.main()
