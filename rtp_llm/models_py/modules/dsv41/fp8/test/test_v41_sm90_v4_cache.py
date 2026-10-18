"""SM90 V4 page-planar codec and FP4-preserving GLOBAL producer contracts."""

import math
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sm90_v4_cache as codec
from rtp_llm.models_py.modules.dsv41.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache as v4_writer,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
    quantize_and_insert_k_cache_fp4,
)
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_flash_mla_cache_format import (
    _read_rows as read_fp4_rows,
)
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_sm90_paged_decode import (
    _read_v4_rows,
)


def _pool(pages, entries, *, cp=1):
    alignment = math.lcm(576, cp)
    stride = math.ceil(entries * 584 / alignment) * alignment + alignment
    raw = torch.full((pages, stride), 0x5A, device="cuda", dtype=torch.uint8)
    return raw, raw.as_strided((pages, entries, 584), (stride, 584, 1))


def _fp4_roundtrip(keys):
    count = keys.shape[0]
    native = torch.zeros((1, count, 288), device=keys.device, dtype=torch.uint8)
    slots = torch.arange(count, device=keys.device)
    quantize_and_insert_k_cache_fp4(keys, native, slots)
    # Independent documented-byte reader, not the new fake-quant implementation.
    return _read_fp4_signed_rows(native, slots).bfloat16()


def _read_fp4_signed_rows(pool, slots):
    # The numerical attention oracle represents both zero codes as +0. For
    # byte-exact V4 RoPE storage, preserve E2M1 code 8 as IEEE negative zero.
    values = read_fp4_rows(pool, slots)
    packed = pool[slots // pool.shape[1], slots % pool.shape[1], :256]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
    return torch.where(codes == 8, -torch.zeros_like(values), values)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90V4CacheTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires an SM90 GPU")
        torch.manual_seed(419221)

    def test_swa_planar_bytes_padding_and_invalid_slots(self):
        for entries in (64, 128, 130):
            with self.subTest(entries=entries):
                raw, pool = _pool(3, entries)
                reference_raw, reference_pool = _pool(3, entries)
                keys = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
                slots = torch.tensor(
                    [entries + 1, 2 * entries + 3, -1, -7, 3 * entries, entries + 7],
                    device="cuda",
                    dtype=torch.int64,
                )
                valid = (slots >= 0) & (slots < 3 * entries)
                safe = torch.where(valid, slots, -1)
                v4_writer(keys, reference_pool, safe)
                codec.quantize_and_insert_swa_k_cache(keys, pool, slots)
                torch.testing.assert_close(raw, reference_raw, rtol=0, atol=0)
                expected = torch.zeros_like(keys, dtype=torch.float32)
                expected[valid] = _read_v4_rows(pool, slots[valid])
                got = codec.dequantize_swa_k_cache(pool, slots, out_dtype=torch.float32)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
                packed = codec.gather_k_cache_bytes(pool, slots)
                self.assertEqual(packed.shape, (6, 584))
                self.assertTrue(bool((packed[~valid] == 0).all()))
                recovered = codec.dequantize_k_cache_bytes(
                    packed, out_dtype=torch.float32
                )
                torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
                # A strided destination must not be rebound or overwrite padding.
                dest = torch.full((6, 544), -123.0, device="cuda")
                returned = codec.dequantize_k_cache_slots(
                    pool, slots, out=dest[:, :512]
                )
                self.assertEqual(returned.data_ptr(), dest.data_ptr())
                torch.testing.assert_close(dest[:, :512], expected, rtol=0, atol=0)
                self.assertTrue(bool((dest[:, 512:] == -123).all()))

    def test_global_preserves_fp4_ties_and_scale_before_v4(self):
        keys = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        # Group max=6 fixes the E4M3 scale to 1, exposing E2M1 halfway ties.
        ties = torch.tensor(
            [
                6,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5,
                -6,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
        keys[0] = ties.repeat(32)
        keys[1] = 0
        keys[2] = ties.repeat(32) * (2**-9)
        keys[3] = ties.repeat(32) * 0.375
        rounded = _fp4_roundtrip(keys)
        torch.testing.assert_close(
            codec.fake_quantize_global_fp4(keys), rounded, rtol=0, atol=0
        )
        raw, pool = _pool(3, 64)
        expected_raw, expected_pool = _pool(3, 64)
        slots = torch.arange(64, 72, device="cuda")
        v4_writer(rounded, expected_pool, slots)
        codec.quantize_and_insert_k_cache(keys, pool, slots)
        torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
        torch.testing.assert_close(
            codec.dequantize_k_cache_slots(pool, slots, out_dtype=torch.float32),
            _read_v4_rows(expected_pool, slots),
            rtol=0,
            atol=0,
        )

    def test_cp_partial_writes_preserve_other_slots_and_fresh_values(self):
        for entries, cp_size in ((128, 3), (130, 5)):
            with self.subTest(entries=entries, cp_size=cp_size):
                expected_raw, expected_pool = _pool(3, entries, cp=cp_size)
                stride = expected_raw.shape[1]
                slices = [part.clone() for part in expected_raw.chunk(cp_size, dim=1)]
                for step in range(2):
                    keys = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
                    slots = torch.tensor(
                        [entries + 2 + step, entries + 7, 2 * entries + 3 + step, -1],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    compact = torch.where(slots >= 0, slots - entries, -1)
                    compaction = SimpleNamespace(
                        unique_blocks=torch.tensor([1, 2], device="cuda"),
                        compact_slots=compact,
                    )
                    fresh_slots = torch.tensor([4, 0, 7, 3], device="cuda")
                    v4_writer(keys, expected_pool, slots)
                    for rank, raw in enumerate(slices):
                        fresh = torch.full(
                            (1, 8, 512), -123.0, device="cuda", dtype=torch.bfloat16
                        )
                        codec.quantize_and_insert_k_cache_cp_byte_sliced(
                            keys,
                            raw,
                            slots,
                            entries,
                            rank,
                            cp_size,
                            compaction,
                            fresh_out=fresh,
                            fresh_slots=fresh_slots,
                        )
                        torch.testing.assert_close(
                            fresh.view(-1, 512)[fresh_slots], keys, rtol=0, atol=0
                        )
                        self.assertTrue(
                            bool((fresh.view(-1, 512)[[1, 2, 5, 6]] == -123).all())
                        )
                    rebuilt = torch.cat(slices, dim=1)
                    self.assertEqual(rebuilt.shape, (3, stride))
                    torch.testing.assert_close(rebuilt, expected_raw, rtol=0, atol=0)

    @torch.no_grad()
    def test_fused_qkv_584_matches_unfused_writer(self):
        from rtp_llm.models_py.modules.dsv41._fused_rmsnorm_rope_triton import (
            fused_rmsnorm_rope,
        )
        from rtp_llm.models_py.modules.dsv41._rope_only_triton import rope_only_inplace
        from rtp_llm.models_py.modules.dsv41.fp8._v41_qkv_rope_cache_triton import (
            try_fused_qkv_rope_cache,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_qkv_rope_cache import (
            make_case,
        )

        for heads in (8, 64):
            with self.subTest(heads=heads):
                case = make_case(2, 2, heads, entries=128)
                raw, case["pool_3d"] = _pool(3, 128)
                reference_raw, reference_pool = _pool(3, 128)
                case["slots"] = torch.tensor([129, -1, 258, 133], device="cuda")
                frequencies = case["freqs_table"][case["positions"].long()]
                expected_q = case["q"].clone()
                rope_only_inplace(expected_q[..., -64:], frequencies)
                expected_kv = fused_rmsnorm_rope(
                    case["kv"], case["kv_norm"], frequencies, 64
                )
                v4_writer(expected_kv.reshape(-1, 512), reference_pool, case["slots"])
                output = try_fused_qkv_rope_cache(**case)
                self.assertIsNotNone(output)
                torch.testing.assert_close(output.q, expected_q, rtol=0, atol=0)
                torch.testing.assert_close(output.kv, expected_kv, rtol=0, atol=0)
                torch.testing.assert_close(raw, reference_raw, rtol=0, atol=0)

    @torch.no_grad()
    def test_fused_prefill_global_584_preserves_fp4_roundtrip(self):
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import rope_only
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_global import (
            _case,
            _compress,
            _latent_reference,
        )

        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                case = _case(32, ratio, batch=2, strided=True)
                raw, case.main_pool = _pool(3, 64)
                expected_raw, expected_pool = _pool(3, 64)
                latent = _latent_reference(case)
                positions = case.positions[case.boundary_indices]
                freqs = case.freqs[positions // ratio * ratio]
                keys = rope_only(latent.clone(), freqs, 64)
                v4_writer(_fp4_roundtrip(keys), expected_pool, case.main_slots)
                actual = _compress(case)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, latent, rtol=0, atol=0)
                torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)

    @torch.no_grad()
    def test_fused_decode_global_584_preserves_fp4_roundtrip(self):
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_global import (
            _candidate,
            _case,
            _clone_case,
        )

        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                native = _case("cuda", ratio, 2, 2)
                candidate = _clone_case(native)
                region = native.owner._global_region()
                pages, entries, _ = native.owner._test_pools[region].shape
                raw, v4_pool = _pool(pages, entries)
                reference_raw, reference_pool = _pool(pages, entries)
                candidate.owner._test_pools[region] = v4_pool
                _candidate(native)
                _candidate(candidate)
                slots = native.owner._slots(region, native.positions, native.req_ids)
                slots = slots[slots >= 0]
                rounded = _read_fp4_signed_rows(
                    native.owner._test_pools[region], slots
                ).bfloat16()
                v4_writer(rounded, reference_pool, slots)
                torch.testing.assert_close(raw, reference_raw, rtol=0, atol=0)
                for other_region, original_pool in native.owner._test_pools.items():
                    if other_region != region:
                        torch.testing.assert_close(
                            candidate.owner._test_pools[other_region],
                            original_pool,
                            rtol=0,
                            atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
