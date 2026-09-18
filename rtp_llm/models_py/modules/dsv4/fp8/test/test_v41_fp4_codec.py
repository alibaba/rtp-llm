"""V4.1-Flash FP4 pool codec against the frozen official TileLang quantizers.

This is the one-time FP4 page-content verification for the migrated pools:
the GLOBAL (288B row-interleaved) and INDEX_K (68B planar) byte layouts are
compared byte-for-byte against the official DeepSeek-V4.1-Flash
``inference/kernel.py`` GPU quantizers, and the pool insert → gather/dequant
roundtrips plus the DeepGEMM MX-mode score contract are exercised end to end.

Requires ``DSV41_MODEL_PATH`` (or ``DSV41_OFFICIAL_KERNEL_PATH``) and a
Blackwell GPU.
"""

import hashlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    FP4_GLOBAL_ENTRY_BYTES,
    FP4_INDEXER_ENTRY_BYTES,
    dequantize_indexer_k_fp4,
    dequantize_k_cache_slots_fp4,
    gather_indexer_k_fp4,
    quantize_and_insert_k_cache_fp4,
    quantize_indexer_k_fp4,
    quantize_rows_fp4,
)

OFFICIAL_KERNEL_SHA256 = (
    "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
)


def _load_official_kernel():
    configured = os.environ.get("DSV41_OFFICIAL_KERNEL_PATH")
    if configured is None:
        model = os.environ.get("DSV41_MODEL_PATH")
        if model is None:
            raise RuntimeError("set DSV41_OFFICIAL_KERNEL_PATH or DSV41_MODEL_PATH")
        configured = str(Path(model) / "inference/kernel.py")
    path = Path(configured)
    if not path.is_file():
        raise RuntimeError(f"missing frozen official GPU quantizer: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != OFFICIAL_KERNEL_SHA256:
        raise RuntimeError(
            f"official quantizer hash mismatch: expected {OFFICIAL_KERNEL_SHA256}, got {actual}"
        )
    # Bazel separates TileLang and z3-solver into different runfiles repositories.
    from rtp_llm.models_py.modules.dsv4.tilelang_kernels import _ensure_libz3_loadable

    _ensure_libz3_loadable()
    spec = importlib.util.spec_from_file_location("_dsv41_frozen_quant_kernel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Importing a reference must not create a pycache beside the source snapshot.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _official_fp4(module, values, group, scale_fp8):
    """Run the official quantizer; return packed (payload, scales) uint8 rows."""
    rows, dimension = values.shape
    scale_dtype = (
        torch.float8_e4m3fn if scale_fp8 else torch.float8_e8m0fnu
    )
    output = torch.empty(
        (rows, dimension // 2), dtype=torch.float4_e2m1fn_x2, device=values.device
    )
    scales = torch.empty(
        (rows, dimension // group), dtype=scale_dtype, device=values.device
    )
    kernel = module.fp4_quant_kernel(
        dimension,
        group,
        in_dtype=module.BF16,
        scale_dtype=module.FP8 if scale_fp8 else module.FE8M0,
    )
    kernel(values, output, scales)
    return output.view(torch.uint8), scales.view(torch.uint8)


def _pool(entries, entry_bytes, blocks, device):
    stride = entries * entry_bytes
    storage = torch.full(
        (blocks, stride), 0x5A, dtype=torch.uint8, device=device
    )
    return storage.as_strided((blocks, entries, entry_bytes), (stride, entry_bytes, 1))


class V41Fp4CodecGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError(
                "this required quantizer test needs an actual Blackwell GPU"
            )
        cls.official = _load_official_kernel()

    def test_global_pool_bytes_match_official_quantizer(self):
        torch.manual_seed(7)
        rows, entries = 257, 128
        values = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        values[0].zero_()
        values[1].fill_(1.5)
        pool = _pool(entries, FP4_GLOBAL_ENTRY_BYTES, 4, values.device)
        slots = torch.arange(rows, dtype=torch.int64, device="cuda") % (
            entries * 3
        ) + entries  # spread over three pages, page 0 untouched
        quantize_and_insert_k_cache_fp4(values, pool, slots)
        expected_payload, expected_scales = _official_fp4(
            self.official, values, 16, True
        )
        for row in range(rows):
            slot = int(slots[row])
            block, offset = slot // entries, slot % entries
            actual = pool[block, offset]
            self.assertEqual(
                tuple(actual[:256].shape), (256,), "payload plane width"
            )
            torch.testing.assert_close(
                actual[:256], expected_payload[row], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual[256:], expected_scales[row], rtol=0, atol=0
            )
        # Untouched page zero keeps its marker bytes.
        self.assertTrue((pool[0] == 0x5A).all())

    def test_indexer_pool_bytes_match_official_quantizer(self):
        torch.manual_seed(11)
        rows, entries = 300, 128
        values = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        values[0].zero_()
        pool = _pool(entries, FP4_INDEXER_ENTRY_BYTES, 4, values.device)
        slots = torch.arange(rows, dtype=torch.int64, device="cuda") % (
            entries * 3
        ) + entries
        quantize_indexer_k_fp4(values, slots, pool)
        expected_payload, expected_scales = _official_fp4(
            self.official, values, 32, False
        )
        raw = pool.view(pool.shape[0], -1)
        for row in range(rows):
            slot = int(slots[row])
            block, offset = slot // entries, slot % entries
            payload = raw[block, offset * 64 : offset * 64 + 64]
            scales = raw[block, entries * 64 + offset * 4 : entries * 64 + offset * 4 + 4]
            torch.testing.assert_close(
                payload, expected_payload[row], rtol=0, atol=0
            )
            torch.testing.assert_close(
                scales, expected_scales[row], rtol=0, atol=0
            )
        self.assertTrue((pool[0] == 0x5A).all())

    def test_global_dequant_matches_official_values(self):
        torch.manual_seed(13)
        rows, entries = 96, 128
        values = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        pool = _pool(entries, FP4_GLOBAL_ENTRY_BYTES, 2, values.device)
        slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        quantize_and_insert_k_cache_fp4(values, pool, slots)
        dequant = dequantize_k_cache_slots_fp4(pool, slots)
        payload, scales = _official_fp4(self.official, values, 16, True)
        expected = _decode_e2m1(payload) * _decode_e4m3_scale(scales).repeat_interleave(
            16, dim=-1
        )
        torch.testing.assert_close(dequant.float(), expected, rtol=0, atol=0)

    def test_indexer_gather_and_dequant_roundtrip(self):
        torch.manual_seed(17)
        rows, entries = 130, 128
        values = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        pool = _pool(entries, FP4_INDEXER_ENTRY_BYTES, 2, values.device)
        slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        quantize_indexer_k_fp4(values, slots, pool)
        payload, sf = gather_indexer_k_fp4(pool, slots)
        expected_payload, expected_scales = _official_fp4(
            self.official, values, 32, False
        )
        torch.testing.assert_close(
            payload.view(torch.uint8), expected_payload, rtol=0, atol=0
        )
        torch.testing.assert_close(
            sf.view(torch.uint8).view(rows, 4), expected_scales, rtol=0, atol=0
        )
        dequant = dequantize_indexer_k_fp4(pool, slots)
        expected = (
            _decode_e2m1(expected_payload)
            * _decode_ue8m0_scale(expected_scales).repeat_interleave(32, dim=-1)
        )
        torch.testing.assert_close(dequant, expected, rtol=0, atol=0)
        # Sentinel slots zero-fill without touching the pool.
        sentinel = torch.full((3,), -1, dtype=torch.int64, device="cuda")
        self.assertTrue((gather_indexer_k_fp4(pool, sentinel)[0] == 0).all())
        self.assertTrue((dequantize_indexer_k_fp4(pool, sentinel) == 0).all())

    def test_row_quant_matches_official_mx_form(self):
        torch.manual_seed(19)
        values = torch.randn(64, 32, 128, device="cuda", dtype=torch.bfloat16)
        payload, sf = quantize_rows_fp4(values)
        expected_payload, expected_scales = _official_fp4(
            self.official, values.reshape(-1, 128), 32, False
        )
        torch.testing.assert_close(
            payload.reshape(-1, 64).view(torch.uint8),
            expected_payload,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sf.reshape(-1).view(torch.uint8).view(-1, 4),
            expected_scales,
            rtol=0,
            atol=0,
        )

    def test_deepgemm_mx_score_matches_reference(self):
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
            fp8_fp4_mqa_indexer_score,
            has_fp8_fp4_mqa_logits,
        )

        if not has_fp8_fp4_mqa_logits():
            self.skipTest("deep_gemm lacks fp8_fp4_mqa_logits")
        torch.manual_seed(23)
        queries, heads, keys = 64, 32, 1000
        q = torch.randn(queries, heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(keys, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(queries, heads, device="cuda") / (heads * 128) ** 0.5
        q_payload, q_sf = quantize_rows_fp4(q)
        k_payload, k_sf = quantize_rows_fp4(k)
        visible = torch.full((queries,), keys, dtype=torch.int32, device="cuda")
        logits = fp8_fp4_mqa_indexer_score(
            q_payload,
            q_sf,
            k_payload,
            k_sf,
            weights,
            torch.zeros_like(visible),
            visible,
            clean_logits=True,
            max_seqlen_k=0,
        )
        q_values = (
            _decode_e2m1(q_payload.reshape(-1, 64).view(torch.uint8))
            * _decode_ue8m0_scale(
                q_sf.reshape(-1).view(torch.uint8).view(-1, 4)
            ).repeat_interleave(32, dim=-1)
        ).reshape(queries, heads, 128)
        k_values = _decode_e2m1(k_payload.view(torch.uint8)) * _decode_ue8m0_scale(
            k_sf.view(torch.uint8).view(-1, 4)
        ).repeat_interleave(32, dim=-1).reshape(keys, 128)
        reference = torch.einsum("mhd,kd->mhk", q_values, k_values)
        reference = (reference.relu_() * weights[:, :, None]).sum(1)
        torch.testing.assert_close(logits, reference, rtol=2e-4, atol=2e-4)


def _decode_e2m1(payload):
    """Decode packed uint8 e2m1 payload rows to float values."""
    low = payload & 15
    high = payload >> 4
    codes = torch.stack((low, high), dim=-1).reshape(payload.shape[0], -1)
    return _e2m1_codes_to_float(codes)


def _e2m1_codes_to_float(codes):
    magnitude = codes & 7
    normal = torch.exp2((magnitude >> 1).float() - 1.0) * (
        1.0 + (magnitude & 1).float() * 0.5
    )
    values = torch.where(magnitude < 2, magnitude.float() * 0.5, normal)
    return torch.where(codes.ge(8), -values, values)


def _decode_e4m3_scale(scales):
    return scales.view(torch.float8_e4m3fn).float()


def _decode_ue8m0_scale(scales):
    exponent = scales.float() - 127.0
    values = torch.exp2(exponent)
    values = torch.where(scales == 0, torch.exp2(torch.tensor(-126.0)), values)
    return torch.where(scales == 255, torch.full_like(values, float("nan")), values)


if __name__ == "__main__":
    unittest.main()