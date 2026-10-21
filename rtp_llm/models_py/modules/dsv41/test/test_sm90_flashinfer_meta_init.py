"""Fresh-process GPU weight-preparation regression; no MoE GEMM is run."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def _worker(experts):
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (9, 0)
    from flashinfer_opensource.fused_moe.prepare import (
        _humming_mxfp4_w4a8_rewrite_lut_cpu,
    )

    from rtp_llm.models_py.modules.dsv41.moe._sm90_flashinfer_moe import (
        Sm90FlashInferLocalExperts,
    )
    from rtp_llm.utils.model_weight import W

    # The subprocess must not have warmed or repaired the lazy upstream cache.
    assert _humming_mxfp4_w4a8_rewrite_lut_cpu.cache_info().currsize == 0
    cfg = SimpleNamespace(
        n_routed_experts=experts,
        ep_size=1,
        dim=128,
        moe_inter_dim=128,
        swiglu_limit=10.0,
    )
    torch.manual_seed(260926)
    raw = {}
    for wk, sk in (
        (W.v4_routed_w1_w, W.v4_routed_w1_s),
        (W.v4_routed_w3_w, W.v4_routed_w3_s),
        (W.v4_routed_w2_w, W.v4_routed_w2_s),
    ):
        raw[wk] = torch.randint(
            0, 256, (experts, 128, 64), device="cuda", dtype=torch.uint8
        ).view(torch.int8)
        raw[sk] = torch.randint(
            121, 125, (experts, 128, 4), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    saved = {key: value.clone() for key, value in raw.items()}
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            local = Sm90FlashInferLocalExperts(cfg)
            consumed = dict(raw)
            local.setup_weights(consumed)
            assert not consumed
            assert torch.empty(()).device.type == "meta"
            assert torch.get_default_dtype() == torch.bfloat16
            lut = _humming_mxfp4_w4a8_rewrite_lut_cpu()
            assert lut.device.type == "cpu" and lut.dtype == torch.uint8
            assert lut.shape == (256, 16)
        assert all(value.is_cuda for value in local.buffers())
        assert local.fc1_residual.dtype == torch.float32
        assert local.fc2_residual.dtype == torch.float32
        assert local.swiglu_limit.dtype == torch.float32
        # Independent unsplit upstream conversion; exercise 96 EP4 experts,
        # the existing 48-expert path, and a partial final chunk.
        from flashinfer_opensource.fused_moe import (
            preprocess_moe_weights_for_sm90_mixed_gemm_humming as prep,
        )

        w31 = torch.cat((raw[W.v4_routed_w3_w], raw[W.v4_routed_w1_w]), dim=1)
        s31 = torch.cat(
            (
                raw[W.v4_routed_w3_s].view(torch.uint8),
                raw[W.v4_routed_w1_s].view(torch.uint8),
            ),
            dim=1,
        )
        fw1, fs1, r1 = prep(w31.view(torch.uint8), s31)
        fw2, fs2, r2 = prep(
            raw[W.v4_routed_w2_w].view(torch.uint8),
            raw[W.v4_routed_w2_s].view(torch.uint8),
        )
        expected = dict(
            fc1_weight=fw1,
            fc1_scale=fs1.view(torch.int32),
            fc1_residual=(r1 * 64.0).contiguous(),
            fc2_weight=fw2,
            fc2_scale=fs2.view(torch.int32),
            fc2_residual=(r2 * 64.0).contiguous(),
        )
        for name, value in expected.items():
            actual = getattr(local, name)
            assert actual.dtype == value.dtype and actual.shape == value.shape
            assert actual.is_contiguous()
            torch.testing.assert_close(
                actual.view(torch.uint8), value.view(torch.uint8), rtol=0, atol=0
            )
        for key in raw:
            torch.testing.assert_close(
                raw[key].view(torch.uint8),
                saved[key].view(torch.uint8),
                rtol=0,
                atol=0,
            )
    finally:
        torch.set_default_dtype(original_dtype)
    torch.cuda.synchronize()
    print("META_INIT_FRESH_PROCESS_GPU_PREP_PASS", flush=True)


@pytest.mark.parametrize("experts", [8, 17, 48, 96])
def test_real_weight_setup_under_meta_and_bf16_in_fresh_process(experts):
    result = subprocess.run(
        [sys.executable, __file__, "--worker", str(experts)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "META_INIT_FRESH_PROCESS_GPU_PREP_PASS" in result.stdout


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]))
    else:
        raise SystemExit(pytest.main([__file__]))
