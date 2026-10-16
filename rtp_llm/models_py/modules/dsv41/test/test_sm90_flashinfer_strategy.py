"""CPU/mock wiring contracts only: these are NOT GPU/collective evidence."""

import dataclasses
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.moe._sm90_flashinfer_moe import (
    Sm90FlashInferLocalExperts,
    validate_lossless_mxfp4_scales,
)
from rtp_llm.models_py.modules.dsv41.moe.strategies.base import MoeCfg, select_strategy
from rtp_llm.models_py.modules.dsv41.moe.strategies import sm90_flashinfer as candidate
from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
    SM90HummingStrategy,
)


def _cfg():
    return MoeCfg(
        layer_id=0,
        dim=5120,
        moe_inter_dim=2304,
        n_routed_experts=384,
        n_activated_experts=6,
        swiglu_limit=10.0,
        ep_size=8,
        ep_rank=0,
        n_local_experts=48,
        local_expert_start=0,
        local_expert_end=48,
        max_tokens_per_rank=1,
        shared_fp8_block_size=32,
        is_decode_role=True,
    )


def test_explicit_selection_preserves_sm90_default(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
    cfg = _cfg()
    assert select_strategy(cfg) is SM90HummingStrategy
    assert (
        select_strategy(cfg, forced="sm90_flashinfer")
        is candidate.SM90FlashInferStrategy
    )
    with pytest.raises(RuntimeError, match="cannot handle"):
        select_strategy(dataclasses.replace(cfg, dim=5121), forced="sm90_flashinfer")
    with pytest.raises(RuntimeError, match="requested strategy"):
        select_strategy(cfg, forced="mega")


class _LocalStub(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.n_local_experts
        self.prepare_calls = []
        self.forward_calls = []

    def prepare(self, rows, topk, *, tune=False):
        self.prepare_calls.append((rows, topk, tune))

    def forward_local(self, x, weights, indices, output_dtype=torch.float32):
        self.forward_calls.append(
            (x.clone(), weights.clone(), indices.clone(), output_dtype)
        )
        # Deliberately no Humming native_a8, w13, w2, or promotion attributes.
        return torch.ones_like(x, dtype=output_dtype)


def _install_wrapper(monkeypatch, cfg, buffer):
    wrapper = SimpleNamespace(
        mode="normal",
        config=SimpleNamespace(ep_size=cfg.ep_size, use_deepep_internode=False),
        buffer=buffer,
    )
    monkeypatch.setitem(
        sys.modules,
        "rtp_llm.models_py.distributed.deepep_wrapper",
        SimpleNamespace(
            DeepEPMode=SimpleNamespace(NORMAL="normal"),
            DeepEPWrapper=SimpleNamespace(_instance=wrapper),
        ),
    )
    return wrapper


def test_static_dispatch_uses_flashinfer_callback_and_unweighted_bf16_combine(
    monkeypatch,
):
    monkeypatch.setenv("DSV41_SM90_DECODE_GRAPH", "1")
    monkeypatch.setenv("DSV41_SM90_DECODE_GRAPH_MAX_ROWS", "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(candidate, "Sm90FlashInferLocalExperts", _LocalStub)
    cfg = _cfg()
    recv_x = torch.randn((8, cfg.dim), dtype=torch.bfloat16)
    recv_ids = torch.tensor([[0, 47, -1, -1, -1, -1, -1, -1]] * 8, dtype=torch.int64)
    recv_weights = torch.rand((8, 8), dtype=torch.float32)
    handle = object()
    buf = SimpleNamespace(
        get_dispatch_layout=Mock(return_value=(1, 2, 3, 4, None)),
        dispatch=Mock(
            return_value=(recv_x, recv_ids, recv_weights, None, handle, None)
        ),
        combine=Mock(side_effect=lambda y, h: (y[:1].clone(), None, None)),
    )
    _install_wrapper(monkeypatch, cfg, buf)
    strategy = candidate.SM90FlashInferStrategy(cfg)
    strategy.setup_runtime()
    assert strategy._local.prepare_calls == [(8, 8, True)]
    x = torch.zeros((1, cfg.dim), dtype=torch.bfloat16)
    weights = torch.full((1, 8), 0.25)
    ids = torch.arange(8, dtype=torch.int64).view(1, 8)
    result = strategy(x, weights, ids)
    assert buf.dispatch.call_args.kwargs["num_worst_tokens"] == 8
    assert buf.dispatch.call_args.kwargs["expert_alignment"] == 1
    local_x, local_w, local_ids, dtype = strategy._local.forward_calls[-1]
    torch.testing.assert_close(local_x, recv_x, rtol=0, atol=0)
    torch.testing.assert_close(local_w, recv_weights, rtol=0, atol=0)
    torch.testing.assert_close(local_ids, recv_ids, rtol=0, atol=0)
    assert dtype is torch.bfloat16
    combine_args = buf.combine.call_args
    assert len(combine_args.args) == 2 and not combine_args.kwargs
    assert combine_args.args[0].dtype is torch.bfloat16
    assert combine_args.args[1] is handle
    assert result.dtype is torch.float32
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)


def test_capture_over_bound_rejected_before_dispatch(monkeypatch):
    monkeypatch.setenv("DSV41_SM90_DECODE_GRAPH", "1")
    monkeypatch.setenv("DSV41_SM90_DECODE_GRAPH_MAX_ROWS", "1")
    monkeypatch.setattr(candidate, "Sm90FlashInferLocalExperts", _LocalStub)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    strategy = candidate.SM90FlashInferStrategy(_cfg())
    with pytest.raises(RuntimeError, match="rows within"):
        strategy(
            torch.zeros((2, 5120), dtype=torch.bfloat16),
            torch.zeros((2, 8)),
            torch.zeros((2, 8), dtype=torch.int64),
        )


def test_lossless_scale_gate_checks_each_expert_and_rejects_clipping():
    def scale(raw):
        return torch.tensor(raw, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    # Large cross-expert differences are legal; range is expert-local.
    validate_lossless_mxfp4_scales(scale([[[20, 31]], [[200, 211]]]))
    for bad in ([[[20, 32]]], [[[0, 1]]], [[[252, 253]]], [[[254, 255]]]):
        with pytest.raises(ValueError, match="requires"):
            validate_lossless_mxfp4_scales(scale(bad))


def test_shared_jit_warmup_accepts_flashinfer_widened_output_contract():
    from rtp_llm.models_py.modules.dsv41.dsv41_kernel_jit_warmup import (
        _shared_expert_signatures,
    )

    strategy = SimpleNamespace(name="sm90_flashinfer")
    moe = SimpleNamespace(
        _shared_executor=SimpleNamespace(name="mxfp8"),
        _routed_includes_shared=False,
        shared_experts=SimpleNamespace(
            w13=SimpleNamespace(K=5120, N=4608),
            w2=SimpleNamespace(K=2304, N=5120),
            swiglu_limit=10.0,
        ),
        _strategy=strategy,
    )
    model = SimpleNamespace(named_modules=lambda: [("layers.0.moe", moe)])
    assert _shared_expert_signatures(model) == {
        (5120, 2304, 10.0, torch.float32, (5120, 1)): "layers.0.moe"
    }


def test_prepare_route_weights_ignore_model_bf16_default(monkeypatch):
    # Exercise the real prepare allocation path on CPU with only CUDA stream
    # metadata and the upstream operator stubbed. No GPU/JIT claim is made.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=-123)
    )
    local = Sm90FlashInferLocalExperts(_cfg())
    local._ready = True
    local.fc1_weight = torch.empty((1,), dtype=torch.uint8)
    local._workspace_size = lambda *args, **kwargs: 256
    local._activation_type = "Swiglu"
    calls = []
    local.forward_local = lambda x, w, ids: calls.append((x, w, ids))
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        local.prepare(8, 8)
        assert torch.get_default_dtype() is torch.bfloat16
    finally:
        torch.set_default_dtype(original)
    assert len(calls) == 1
    x, weights, indices = calls[0]
    assert x.dtype is torch.bfloat16
    assert weights.dtype is torch.float32
    assert indices.dtype is torch.int64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
