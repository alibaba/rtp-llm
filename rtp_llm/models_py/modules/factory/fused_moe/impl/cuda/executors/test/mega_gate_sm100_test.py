"""SM100/SM103 GPU parity, CUDA graph barrier, and MegaMoe x-pack execution tests."""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    mega_gate as mega_gate_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.mega_gate import (
    can_use_bf16_mega_gate,
    is_bf16_mega_gate_available,
    prepare_capture_stream_barrier,
    run_bf16_mega_gate,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    stage_shared_fp8_input_scales,
)
from rtp_llm.utils.model_weight import W


@contextmanager
def _env(**pairs: str):
    saved = {k: os.environ.get(k) for k in pairs}
    try:
        for k, v in pairs.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_gate(
    *,
    hidden: int = 512,
    experts: int = 16,
    topk: int = 4,
    score_func: str = "sqrtsoftplus",
    is_hash: bool = False,
    with_bias_vl: bool = False,
    has_visual_tokens: bool = False,
    vocab_size: int = 128,
    route_scale: float = 2.5,
) -> Gate:
    device = torch.device("cuda:0")
    layer_weights = {
        W.moe_gate: torch.randn((experts, hidden), dtype=torch.bfloat16, device=device) * 0.1,
        W.moe_gate_bias: torch.randn((experts,), dtype=torch.float32, device=device) * 0.05,
        W.moe_gate_tid2eid: torch.stack(
            [torch.randperm(experts, device=device)[:topk] for _ in range(vocab_size)]
        ).contiguous(),
    }
    if with_bias_vl:
        layer_weights[W.moe_gate_bias_vl] = (
            torch.randn((experts,), dtype=torch.float32, device=device) * 0.05
        )
    gate = Gate(
        layer_id=0,
        dim=hidden,
        n_routed_experts=experts,
        n_activated_experts=topk,
        score_func=score_func,
        route_scale=route_scale,
        n_hash_layers=1 if is_hash else 0,
        vocab_size=vocab_size,
        layer_weights=layer_weights,
    )
    gate._has_visual_tokens = has_visual_tokens
    return gate


class _DummySymBuf:
    def __init__(self, max_tokens: int, hidden: int, topk: int, block_m: int = 128):
        device = torch.device("cuda:0")
        self.num_max_tokens_per_rank = max_tokens
        self.x = torch.empty((max_tokens, hidden), dtype=torch.float8_e4m3fn, device=device)
        self.x_sf = torch.empty((max_tokens, hidden // 128), dtype=torch.int32, device=device)
        self.topk_idx = torch.empty((max_tokens, topk), dtype=torch.int64, device=device)
        self.topk_weights = torch.empty((max_tokens, topk), dtype=torch.float32, device=device)
        num_blocks = (max_tokens + block_m - 1) // block_m
        aligned = ((block_m + 127) // 128) * 128
        self.shared_l1_acts_sf = torch.empty(
            (num_blocks * aligned, hidden // 128), dtype=torch.int32, device=device
        )


class MegaGateSm100Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required by SM100 target")
        if torch.cuda.get_device_capability()[0] != 10:
            raise AssertionError("SM100/SM103 is required")
        if not is_bf16_mega_gate_available():
            raise AssertionError("deep_gemm.bf16_mega_gate is required on CUDA13 SM100")

    def test_gate_parity_nonhash_and_hash_vl(self):
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        for score_func in ("sqrtsoftplus", "sigmoid"):
            for tokens in (1, 7, 17, 64):
                with self.subTest(score_func=score_func, tokens=tokens):
                    gate = _make_gate(
                        hidden=512,
                        experts=256,
                        topk=6,
                        score_func=score_func,
                        route_scale=2.5,
                    )
                    x = torch.randn((tokens, 512), dtype=torch.bfloat16, device=device)
                    self.assertTrue(can_use_bf16_mega_gate(gate, x))
                    with _env(MOE_GATE_FP32="1"):
                        ref_w, ref_i = gate(x)
                    got_w, got_i = gate(x)
                    self.assertEqual(got_i.dtype, torch.int64)
                    self.assertEqual(got_w.dtype, torch.float32)
                    # Sorted expert sets match closely across bf16->fp32 accumulation vs bf16->fp32 casts.
                    for row in range(tokens):
                        self.assertEqual(set(ref_i[row].tolist()), set(got_i[row].tolist()))
                    torch.testing.assert_close(
                        got_w.sort(dim=-1).values,
                        ref_w.sort(dim=-1).values,
                        rtol=2e-2,
                        atol=5e-3,
                    )

        # Hash + VL mixed batch (text + image tokens) preserves tid2eid and matches eager
        hgate = _make_gate(
            hidden=512,
            experts=64,
            topk=6,
            score_func="sqrtsoftplus",
            is_hash=True,
            with_bias_vl=True,
            has_visual_tokens=True,
            vocab_size=64,
            route_scale=2.5,
        )
        tid2eid_orig = hgate.tid2eid.clone()
        x_vl = torch.randn((8, 512), dtype=torch.bfloat16, device=device)
        input_ids = torch.tensor([1, 70, 5, 80, 11, 90, 3, 64], dtype=torch.long, device=device)
        with _env(MOE_GATE_FP32="1"):
            ref_w, ref_i = hgate(x_vl, input_ids)
        got_w, got_i = hgate(x_vl, input_ids)
        self.assertTrue(torch.equal(hgate.tid2eid, tid2eid_orig))
        # Text tokens (even indices) have deterministic tid2eid
        for row in (0, 2, 4, 6):
            self.assertTrue(torch.equal(got_i[row], ref_i[row]))
        # Image tokens (odd indices) select the same expert sets
        for row in (1, 3, 5, 7):
            self.assertEqual(set(got_i[row].tolist()), set(ref_i[row].tolist()))
        torch.testing.assert_close(
            got_w.sort(dim=-1).values,
            ref_w.sort(dim=-1).values,
            rtol=2e-2,
            atol=5e-3,
        )

        # Small E=16, large H=4096 (high split-K), T=2, hash=True without VL (matches 2-rank UT shape)
        hgate_small_e = _make_gate(
            hidden=4096,
            experts=16,
            topk=4,
            score_func="sqrtsoftplus",
            is_hash=True,
            with_bias_vl=False,
            has_visual_tokens=False,
            vocab_size=64,
            route_scale=1.5,
        )
        x_small = torch.randn((2, 4096), dtype=torch.bfloat16, device=device)
        ids_small = torch.tensor([1, 0], dtype=torch.long, device=device)
        with _env(MOE_GATE_FP32="1"):
            ref_ws, ref_is = hgate_small_e(x_small, ids_small)
        got_ws, got_is = hgate_small_e(x_small, ids_small)
        self.assertTrue(torch.equal(got_is, ref_is))
        torch.testing.assert_close(got_ws, ref_ws, rtol=5e-2, atol=2e-2)

    def test_cuda_graph_capture_and_replay_on_dedicated_stream(self):
        torch.manual_seed(7)
        device = torch.device("cuda:0")
        gate = _make_gate(hidden=512, experts=64, topk=6, score_func="sqrtsoftplus")
        x = torch.randn((4, 512), dtype=torch.bfloat16, device=device)
        # Ensure _LAST_SPEC_BY_DEVICE has recorded spec on default stream
        w_eager, i_eager = gate(x)

        capture_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(capture_stream):
            # Capture before barrier init cleanly falls back; after barrier init uses bf16_mega_gate
            prepared = prepare_capture_stream_barrier(device)
            self.assertTrue(prepared)

            x_static = x.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=capture_stream):
                # Inside capture, canonical bf16_mega_gate runs because capture_stream is warmed
                self.assertTrue(can_use_bf16_mega_gate(gate, x_static))
                w_cap, i_cap = gate(x_static)
            x_static.copy_(x)
            g.replay()
            torch.cuda.current_stream(device).synchronize()

        self.assertTrue(torch.equal(i_cap, i_eager))
        torch.testing.assert_close(w_cap, w_eager, rtol=1e-5, atol=1e-5)

        # Unwarmed dedicated stream inside capture must fall back cleanly without cuBLAS error.
        with mock.patch.object(
            mega_gate_module, "can_use_bf16_mega_gate", return_value=False
        ):
            w_fb_ref, i_fb_ref = gate(x)

        unwarmed_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(unwarmed_stream):
            x_fb = x.clone()
            g_fb = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_fb, stream=unwarmed_stream):
                self.assertFalse(can_use_bf16_mega_gate(gate, x_fb))
                w_fb, i_fb = gate(x_fb)
            w_fb.fill_(float("nan"))
            i_fb.fill_(-1)
            x_fb.copy_(x)
            g_fb.replay()
            torch.cuda.current_stream(device).synchronize()
        self.assertTrue(torch.equal(i_fb, i_fb_ref))
        torch.testing.assert_close(w_fb, w_fb_ref, rtol=1e-5, atol=1e-5)
        for row in range(x.size(0)):
            self.assertEqual(set(i_fb[row].tolist()), set(i_eager[row].tolist()))
        torch.testing.assert_close(
            w_fb.sort(dim=-1).values,
            w_eager.sort(dim=-1).values,
            rtol=2e-2,
            atol=5e-3,
        )

    def test_mega_moe_se_forward_mega_gate_matches_gate_pack_staging(self):
        torch.manual_seed(19)
        device = torch.device("cuda:0")
        tokens, hidden, experts, topk, block_m = 17, 512, 64, 6, 128

        ex = MegaMoeSEExecutor.__new__(MegaMoeSEExecutor)
        torch.nn.Module.__init__(ex)
        ex.cfg = SimpleNamespace(layer_id=0, n_activated_experts=topk, dim=hidden)
        ex._input_packer = FusedMegaMoEInputPacker()
        ex._mega_y = torch.empty((32, hidden), dtype=torch.bfloat16, device=device)
        ex._shared_mid_fp8 = torch.empty((32, 128), dtype=torch.float8_e4m3fn, device=device)
        ex._block_m = lambda t: block_m
        ex._check_shared_expert_ready = lambda: None
        ex._shared_fp8_scale_stage_target = lambda t: (
            ex._mega_buf.shared_l1_acts_sf,
            block_m,
        )

        ref_buf = _DummySymBuf(32, hidden, topk, block_m=block_m)
        got_buf = _DummySymBuf(32, hidden, topk, block_m=block_m)
        ex._mega_buf = got_buf

        gate = _make_gate(hidden=hidden, experts=experts, topk=topk, score_func="sqrtsoftplus")
        x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
        out_buf = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)

        with mock.patch.object(ex, "_launch") as launch_mock:
            y, got_w, got_i = ex.forward_mega_gate(x, gate, out=out_buf)
            launch_mock.assert_called_once_with(out_buf, tokens, device)
        self.assertIs(y, out_buf)

        # Reference: standalone run_bf16_mega_gate + FusedMegaMoEInputPacker + stage_shared_fp8_input_scales
        ref_i, ref_w = run_bf16_mega_gate(gate, x)
        FusedMegaMoEInputPacker().pack(x, ref_w, ref_i, ref_buf, tokens)
        ref_shared_sf = torch.empty_like(got_buf.shared_l1_acts_sf)
        stage_shared_fp8_input_scales(
            ref_buf.x_sf[:tokens],
            ref_shared_sf,
            tokens=tokens,
            block_m=block_m,
        )

        self.assertTrue(torch.equal(got_i, ref_i))
        torch.testing.assert_close(got_w, ref_w, rtol=0, atol=0)
        self.assertTrue(torch.equal(got_buf.topk_idx[:tokens], ref_i))
        self.assertTrue(torch.equal(got_buf.topk_weights[:tokens], ref_w))
        self.assertTrue(
            torch.equal(
                got_buf.x[:tokens].view(torch.uint8),
                ref_buf.x[:tokens].view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(got_buf.x_sf[:tokens], ref_buf.x_sf[:tokens]))
        valid_rows = ((tokens + block_m - 1) // block_m) * block_m
        self.assertTrue(
            torch.equal(
                got_buf.shared_l1_acts_sf[:valid_rows],
                ref_shared_sf[:valid_rows],
            )
        )


if __name__ == "__main__":
    unittest.main()
