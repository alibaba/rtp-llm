"""Native V4.1 Q/KV preprocessing and 528-byte cache equivalence."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4._rope_only_triton import rope_only_inplace
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_qkv_rope_cache_triton as fused
from rtp_llm.models_py.modules.dsv4.fp8._v41_swa_triton import (
    quantize_and_insert_swa_k_cache,
)


def make_case(
    batch=4,
    width=6,
    heads=64,
    *,
    entries=134,
    strided=True,
    device="cuda",
    block_stride=None
):
    tokens = batch * width
    pages = max(2, (tokens + entries - 1) // entries)
    row_width = 1792 if strided else 512
    projected = torch.randn(
        batch, width, row_width, device=device, dtype=torch.bfloat16
    )
    kv = projected[..., -512:]
    stride = entries * 528 + 128 if block_stride is None else block_stride
    raw = torch.empty(
        (pages - 1) * stride + entries * 528, device=device, dtype=torch.uint8
    )
    pool = raw.as_strided((pages, entries, 528), (stride, 528, 1))
    pool.fill_(173)
    angle = torch.randn(1031, 32, device=device)
    return dict(
        q=torch.randn(batch, width, heads, 512, device=device, dtype=torch.bfloat16),
        kv=kv,
        kv_norm=(1 + torch.randn(512, device=device) * 0.1).bfloat16(),
        positions=torch.arange(tokens, device=device, dtype=torch.int32) * 7,
        freqs_table=torch.polar(torch.ones_like(angle), angle),
        pool_3d=pool,
        slots=torch.arange(tokens, device=device, dtype=torch.int64),
    )


def reference(case):
    q = case["q"].clone()
    frequencies = case["freqs_table"].index_select(0, case["positions"].long())
    rope_only_inplace(q[..., -64:], frequencies)
    kv = fused_rmsnorm_rope(case["kv"], case["kv_norm"], frequencies, 64)
    pool = torch.empty_strided(
        case["pool_3d"].shape,
        case["pool_3d"].stride(),
        device=q.device,
        dtype=torch.uint8,
    )
    pool.copy_(case["pool_3d"])
    quantize_and_insert_swa_k_cache(kv.reshape(-1, 512), pool, case["slots"])
    return q, kv, frequencies, pool


class V41QKVRopeCacheCPUTest(unittest.TestCase):
    def test_cpu_and_disabled_inputs_have_no_side_effects(self):
        case = make_case(1, 1, 1, device="cpu")
        q, kv, pool = case["q"].clone(), case["kv"].clone(), case["pool_3d"].clone()
        self.assertFalse(fused.is_supported(**case))
        self.assertIsNone(fused.try_fused_qkv_rope_cache(**case))
        with patch.dict(os.environ, {"DSV41_FUSED_QKV_ROPE_CACHE": "0"}):
            self.assertIsNone(fused.try_fused_qkv_rope_cache(**case))
        for got, expected in (
            (case["q"], q),
            (case["kv"], kv),
            (case["pool_3d"], pool),
        ):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41QKVRopeCacheCudaTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {"DSV41_FUSED_QKV_ROPE_CACHE": "1", "DSV4_TRAP_INVALID_KV_ACCESS": "1"},
        )
        env.start()
        self.addCleanup(env.stop)
        torch.manual_seed(417)

    def check_case(self, case, gathered=False):
        expected_q, expected_kv, expected_freqs, expected_pool = reference(case)
        protected = {
            name: case[name].clone()
            for name in ("kv", "kv_norm", "positions", "freqs_table", "slots")
        }
        kwargs = {"freqs_cis": expected_freqs} if gathered else {}
        output = fused.try_fused_qkv_rope_cache(**case, **kwargs)
        self.assertIsNotNone(output)
        self.assertIs(output.q, case["q"])
        if gathered:
            self.assertIs(output.freqs_cis, expected_freqs)
        torch.testing.assert_close(output.q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(output.kv, expected_kv, rtol=0, atol=0)
        torch.testing.assert_close(output.freqs_cis, expected_freqs, rtol=0, atol=0)
        torch.testing.assert_close(case["pool_3d"], expected_pool, rtol=0, atol=0)
        for name, previous in protected.items():
            torch.testing.assert_close(case[name], previous, rtol=0, atol=0)
        return output

    @torch.no_grad()
    def test_decode_shapes_strides_heads_and_masked_slots(self):
        for batch, width, heads, entries, strided in (
            (1, 1, 1, 2, False),
            (1, 5, 7, 17, True),
            (4, 5, 64, 134, True),
            (4, 6, 64, 134, True),
            (8, 6, 64, 256, False),
        ):
            for gathered in (False, True):
                with self.subTest(
                    batch=batch,
                    width=width,
                    heads=heads,
                    entries=entries,
                    strided=strided,
                    gathered=gathered,
                ):
                    case = make_case(
                        batch, width, heads, entries=entries, strided=strided
                    )
                    case["slots"][0] = -1
                    if case["slots"].numel() > 1:
                        case["slots"][-1] = -7
                    self.check_case(case, gathered)

    @torch.no_grad()
    def test_zero_and_large_norm_input(self):
        for scale in (0.0, 1e-5, 32.0):
            for heads in (8, 64):
                for gathered in (False, True):
                    with self.subTest(scale=scale, heads=heads, gathered=gathered):
                        case = make_case(4, 6, heads)
                        case["kv"].mul_(scale)
                        self.check_case(case, gathered)

    @torch.no_grad()
    def test_layout_gate_584_and_invalid_metadata(self):
        case = make_case(1, 1, 8)
        before = case["q"].clone()
        variants = (
            {
                **case,
                "pool_3d": torch.empty(2, 134, 584, device="cuda", dtype=torch.uint8),
            },
            {**case, "q": case["q"].float()},
            {**case, "kv_norm": case["kv_norm"].float()},
            {**case, "slots": case["slots"].float()},
            {**case, "positions": case["positions"].float()},
            {**case, "freqs_table": case["freqs_table"].to(torch.complex128)},
        )
        for variant in variants:
            self.assertFalse(fused.is_supported(**variant))
            self.assertIsNone(fused.try_fused_qkv_rope_cache(**variant))
        torch.testing.assert_close(case["q"], before, rtol=0, atol=0)

    @torch.no_grad()
    def test_empty_batch(self):
        case = make_case(0, 6, 64)
        output = fused.try_fused_qkv_rope_cache(**case)
        self.assertIsNotNone(output)
        self.assertEqual(output.kv.shape, (0, 6, 512))
        self.assertEqual(output.freqs_cis.shape, (0, 32))

    @torch.no_grad()
    def test_cache_offset_above_int32(self):
        case = make_case(1, 1, 8, entries=2, block_stride=2**31 + 128)
        case["slots"].fill_(2)
        self.check_case(case)

    @torch.no_grad()
    def test_nondefault_stream_graph_replays_updated_positions_and_slots(self):
        case = make_case(4, 6, 64)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused.try_fused_qkv_rope_cache(**case)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = fused.try_fused_qkv_rope_cache(**case)
        self.assertIsNotNone(actual)
        for delta in (1, 7, 17):
            case["q"].copy_(torch.randn_like(case["q"]))
            case["kv"].copy_(torch.randn_like(case["kv"]))
            case["positions"].add_(delta)
            case["slots"].add_(1)
            case["pool_3d"].fill_(173)
            q, kv, freqs, pool = reference(case)
            graph.replay()
            torch.testing.assert_close(actual.q, q, rtol=0, atol=0)
            torch.testing.assert_close(actual.kv, kv, rtol=0, atol=0)
            torch.testing.assert_close(actual.freqs_cis, freqs, rtol=0, atol=0)
            torch.testing.assert_close(case["pool_3d"], pool, rtol=0, atol=0)

    def test_invalid_positive_slot_traps_in_subprocess(self):
        code = """import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import torch
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_qkv_rope_cache import make_case, fused
case = make_case(1, 1, 8)
case['slots'].fill_(case['pool_3d'].shape[0] * case['pool_3d'].shape[1])
with torch.no_grad():
    fused.try_fused_qkv_rope_cache(**case)
torch.cuda.synchronize()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env=os.environ.copy(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(
            result.stderr,
            "illegal instruction|device-side assert|unspecified launch failure",
        )


if __name__ == "__main__":
    unittest.main()
