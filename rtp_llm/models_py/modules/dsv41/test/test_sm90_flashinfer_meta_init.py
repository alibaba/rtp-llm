"""Fresh-process GPU weight-preparation regression; no MoE GEMM is run."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def _worker():
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
        n_routed_experts=8,
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
            0, 256, (8, 128, 64), device="cuda", dtype=torch.uint8
        ).view(torch.int8)
        raw[sk] = torch.randint(
            121, 125, (8, 128, 4), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
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
        reference = Sm90FlashInferLocalExperts(cfg)
        reference.setup_weights(dict(raw))
        expected = dict(reference.named_buffers())
        for name, value in local.named_buffers():
            assert value.dtype == expected[name].dtype
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    finally:
        torch.set_default_dtype(original_dtype)
    torch.cuda.synchronize()
    print("META_INIT_FRESH_PROCESS_GPU_PREP_PASS", flush=True)


def test_real_weight_setup_under_meta_and_bf16_in_fresh_process():
    result = subprocess.run(
        [sys.executable, __file__, "--worker"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "META_INIT_FRESH_PROCESS_GPU_PREP_PASS" in result.stdout


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker()
    else:
        raise SystemExit(pytest.main([__file__]))
