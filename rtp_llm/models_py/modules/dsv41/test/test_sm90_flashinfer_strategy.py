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
from rtp_llm.models_py.modules.dsv41.moe.strategies import base as strategy_base
from rtp_llm.models_py.modules.dsv41.moe.strategies import sm90_flashinfer as candidate
from rtp_llm.models_py.modules.dsv41.moe.strategies.base import MoeCfg, select_strategy


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


def test_sm90_defaults_to_flashinfer_and_rejects_retired_backends(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
    cfg = _cfg()
    assert select_strategy(cfg) is candidate.SM90FlashInferStrategy
    assert (
        select_strategy(cfg, forced="mega", strict=False)
        is candidate.SM90FlashInferStrategy
    )
    with pytest.raises(RuntimeError, match="requested strategy"):
        select_strategy(cfg, forced="sm90_humming")
    assert (
        select_strategy(cfg, forced="sm90_flashinfer")
        is candidate.SM90FlashInferStrategy
    )
    with pytest.raises(RuntimeError, match="cannot handle"):
        select_strategy(dataclasses.replace(cfg, dim=5121), forced="sm90_flashinfer")
    with pytest.raises(RuntimeError, match="requested strategy"):
        select_strategy(cfg, forced="mega")


@pytest.mark.parametrize(
    "hip_version, capability",
    [(None, (10, 0)), (None, (9, 4)), ("6.4", (9, 0)), ("6.4", (9, 4))],
    ids=["sm100", "non_sm90_cuda", "hip_9_0", "hip_9_4"],
)
def test_non_sm90_keeps_existing_selector(monkeypatch, hip_version, capability):
    monkeypatch.setenv("DSV41_MEGA_SHARED_EXPERT", "1")
    monkeypatch.setenv("DSV4_USE_MEGA_MOE_SE", "1")
    monkeypatch.setenv("DSV4_USE_MEGA_MOE_FUSED", "0")
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    # Stub kernel availability, not architecture routing: non-SM90 devices
    # must reach the original selector, without importing CUDA FlashInfer.
    existing = SimpleNamespace(name="mega_se", can_handle=lambda cfg: True)
    legacy = SimpleNamespace(name="mega", can_handle=lambda cfg: True)
    monkeypatch.setattr(
        strategy_base,
        "_STRATEGY_PRIORITY",
        [existing, legacy, candidate.SM90FlashInferStrategy],
    )
    assert not candidate.SM90FlashInferStrategy.can_handle(_cfg())
    assert select_strategy(_cfg()) is existing
    assert select_strategy(_cfg(), forced="mega") is existing
    with pytest.raises(RuntimeError, match="cannot handle cfg"):
        select_strategy(_cfg(), forced="sm90_flashinfer")
    monkeypatch.setenv("DSV41_MEGA_SHARED_EXPERT", "0")
    monkeypatch.setenv("DSV4_USE_MEGA_MOE_SE", "0")
    assert select_strategy(_cfg(), forced="mega") is legacy


class _LocalStub(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.n_local_experts
        self.prepare_calls = []
        self.forward_calls = []
        self.eager_calls = []

    def prepare(self, rows, topk, *, tune=False):
        self.prepare_calls.append((rows, topk, tune))

    def forward_local(
        self, x, weights, indices, output_dtype=torch.float32, *, recv_count=None
    ):
        self.forward_calls.append(
            (x.clone(), weights.clone(), indices.clone(), output_dtype, recv_count)
        )
        # No Humming native_a8, w13, w2, or promotion attributes.
        return torch.ones_like(x, dtype=output_dtype)

    def _forward_into_buf(self, x, weights, indices, local_start, local_end):
        self.eager_calls.append(
            (x.clone(), weights.clone(), indices.clone(), local_start, local_end)
        )
        return torch.ones_like(x, dtype=torch.float32)


def _install_v2(monkeypatch, cfg, *, graph, recv_rows=None):
    monkeypatch.setenv("DSV41_SM90_MOE_COMM", "v2")
    monkeypatch.setenv("DSV41_SM90_V2_NUM_SMS", "20")
    monkeypatch.setenv("DSV41_SM90_DECODE_GRAPH", "1" if graph else "0")
    monkeypatch.setenv("DSV41_SM90_DECODE_GRAPH_MAX_ROWS", "1")
    monkeypatch.setattr(candidate, "Sm90FlashInferLocalExperts", _LocalStub)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=132),
    )
    group = SimpleNamespace(size=lambda: cfg.ep_size, rank=lambda: cfg.ep_rank)
    legacy = SimpleNamespace(group=group, dispatch=Mock(), combine=Mock())
    wrapper = SimpleNamespace(
        mode="normal",
        config=SimpleNamespace(ep_size=cfg.ep_size, use_deepep_internode=False),
        buffer=legacy,
    )
    monkeypatch.setitem(
        sys.modules,
        "rtp_llm.models_py.distributed.deepep_wrapper",
        SimpleNamespace(
            DeepEPMode=SimpleNamespace(NORMAL="normal"),
            DeepEPWrapper=SimpleNamespace(_instance=wrapper),
        ),
    )
    # CPU/mock transport only: graph uses its fixed receive capacity, eager
    # uses actual rows. The device scalar's identity must reach local FI.
    if recv_rows is None:
        recv_rows = cfg.ep_size if graph else 3
    recv_x = torch.randn((recv_rows, cfg.dim), dtype=torch.bfloat16)
    recv_ids = torch.tensor(
        [[0, cfg.n_local_experts - 1, -1, -1, -1, -1, -1, -1]] * recv_rows
    )
    recv_weights = torch.rand((recv_rows, 8), dtype=torch.float32)
    count = torch.tensor(3, dtype=torch.int32)
    handle = SimpleNamespace(psum_num_recv_tokens_per_scaleup_rank=[count])
    buf = SimpleNamespace(
        num_bytes=1234,
        dispatch=Mock(return_value=(recv_x, recv_ids, recv_weights, handle, None)),
        combine=Mock(side_effect=lambda y, h: (y[:1].clone(), None, None)),
    )
    factory = Mock(return_value=buf)
    monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(ElasticBuffer=factory))
    from rtp_llm.models_py.modules.dsv41.moe import _sm90_dispatch_pad

    def pad(indices, weights, pad_to):
        return (
            torch.cat(
                (
                    indices,
                    indices.new_full((indices.shape[0], pad_to - indices.shape[1]), -1),
                ),
                1,
            ),
            torch.cat(
                (
                    weights,
                    weights.new_zeros((weights.shape[0], pad_to - weights.shape[1])),
                ),
                1,
            ),
        )

    padding = Mock(side_effect=pad)
    monkeypatch.setattr(_sm90_dispatch_pad, "pad_deepep_routes", padding)
    return wrapper, factory, buf, (recv_x, recv_ids, recv_weights, handle), padding


def _inputs(rows=1):
    return (
        torch.zeros((rows, 5120), dtype=torch.bfloat16),
        torch.full((rows, 6), 1.0 / 6),
        torch.arange(6, dtype=torch.int64).expand(rows, 6),
    )


@pytest.mark.parametrize("ep_size", [4, 8])
@pytest.mark.parametrize("graph", [False, True])
def test_v2_group_shared_buffer_and_prefill_decode_contracts(
    monkeypatch, graph, ep_size
):
    cfg = dataclasses.replace(
        _cfg(),
        ep_size=ep_size,
        n_local_experts=384 // ep_size,
        local_expert_end=384 // ep_size,
        max_tokens_per_rank=8192,
        is_decode_role=graph,
    )
    wrapper, factory, buf, received, padding = _install_v2(
        monkeypatch, cfg, graph=graph
    )
    strategy = candidate.SM90FlashInferStrategy(cfg)
    strategy.setup_runtime()
    # Repeated setup and another layer reuse one transport allocation/group.
    strategy.setup_runtime()
    another = candidate.SM90FlashInferStrategy(dataclasses.replace(cfg, layer_id=1))
    another.setup_runtime()
    capacity = 1 if graph else 8192
    factory.assert_called_once_with(
        wrapper.buffer.group, num_max_tokens_per_rank=capacity, hidden=5120, num_topk=8
    )
    assert strategy._comm_buffer is another._comm_buffer is buf
    assert list(wrapper._dsv41_elastic_buffers.values()) == [buf]
    assert strategy._local.prepare_calls == ([(ep_size, 8, True)] * 2 if graph else [])
    result = strategy(*_inputs())
    padding.assert_called_once()
    assert padding.call_args.kwargs == {"pad_to": 8}
    call = buf.dispatch.call_args
    assert len(call.args) == 1
    assert call.kwargs["num_experts"] == 384
    assert call.kwargs["num_max_tokens_per_rank"] == capacity
    assert call.kwargs["expert_alignment"] == 1
    assert call.kwargs["num_sms"] == 20
    assert call.kwargs["do_cpu_sync"] is (not graph)
    assert call.kwargs["do_expand"] is False
    assert call.kwargs["topk_idx"].shape == (1, 8)
    assert torch.equal(call.kwargs["topk_idx"][:, 6:], torch.full((1, 2), -1))
    assert torch.equal(call.kwargs["topk_weights"][:, 6:], torch.zeros((1, 2)))
    recv_x, recv_ids, recv_weights, handle = received
    if graph:
        local_x, local_w, local_ids, dtype, count = strategy._local.forward_calls[-1]
        assert dtype is torch.bfloat16
        assert count is handle.psum_num_recv_tokens_per_scaleup_rank[-1]
        assert not strategy._local.eager_calls
    else:
        local_x, local_w, local_ids, start, end = strategy._local.eager_calls[-1]
        assert (start, end) == (0, cfg.n_local_experts)
        assert not strategy._local.forward_calls
    for actual, expected in (
        (local_x, recv_x),
        (local_w, recv_weights),
        (local_ids, recv_ids),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    args = buf.combine.call_args
    assert len(args.args) == 2 and not args.kwargs
    assert args.args[0].dtype is torch.bfloat16 and args.args[1] is handle
    assert result.dtype is torch.float32
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)
    wrapper.buffer.dispatch.assert_not_called()
    wrapper.buffer.combine.assert_not_called()


@pytest.mark.parametrize("graph", [False, True])
def test_v2_guards_reject_before_dispatch(monkeypatch, graph):
    cfg = dataclasses.replace(_cfg(), max_tokens_per_rank=1, is_decode_role=graph)
    _, _, buf, _, _ = _install_v2(monkeypatch, cfg, graph=graph)
    strategy = candidate.SM90FlashInferStrategy(cfg)
    with pytest.raises(RuntimeError, match="runtime has not been initialized"):
        strategy(*_inputs())
    strategy.setup_runtime()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="exceeds configured SM90 V2 token capacity"):
        strategy(*_inputs(2))
    if not graph:
        with pytest.raises(RuntimeError, match="require eager execution"):
            strategy(*_inputs())
    buf.dispatch.assert_not_called()


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("draft_first", [False, True])
def test_v2_target_draft_buffers_are_isolated_and_reused(
    monkeypatch, graph, draft_first
):
    target_cfg = dataclasses.replace(
        _cfg(), max_tokens_per_rank=8192, is_decode_role=graph
    )
    draft_cfg = dataclasses.replace(
        target_cfg,
        n_routed_experts=128,
        n_activated_experts=3,
        n_local_experts=16,
        local_expert_end=16,
    )
    wrapper, factory, _, _, _ = _install_v2(monkeypatch, target_cfg, graph=graph)
    factory.side_effect = lambda *args, **kwargs: SimpleNamespace(num_bytes=1234)
    configs = [draft_cfg, target_cfg] if draft_first else [target_cfg, draft_cfg]
    first, second = [candidate.SM90FlashInferStrategy(cfg) for cfg in configs]
    first.setup_runtime()
    original = first._comm_buffer
    second.setup_runtime()
    assert first._comm_buffer is original
    assert second._comm_buffer is not original
    assert factory.call_count == 2
    for cfg, expected in zip(configs, [first, second]):
        another = candidate.SM90FlashInferStrategy(dataclasses.replace(cfg, layer_id=1))
        another.setup_runtime()
        expected.setup_runtime()
        assert another._comm_buffer is expected._comm_buffer
    assert factory.call_count == 2
    assert len(wrapper._dsv41_elastic_buffers) == 2
    if graph:
        for strategy in [first, second]:
            topk = 8 if strategy.cfg.n_routed_experts == 384 else 4
            assert strategy._local.prepare_calls == [(8, topk, True)] * 2


def test_v2_distinct_capacities_keep_existing_buffer_and_group_guard(monkeypatch):
    cfg = dataclasses.replace(_cfg(), max_tokens_per_rank=8192, is_decode_role=False)
    wrapper, factory, _, _, _ = _install_v2(monkeypatch, cfg, graph=False)
    factory.side_effect = lambda *args, **kwargs: SimpleNamespace(num_bytes=1234)
    original = candidate.SM90FlashInferStrategy(cfg)
    original.setup_runtime()
    smaller = candidate.SM90FlashInferStrategy(
        dataclasses.replace(cfg, max_tokens_per_rank=4096)
    )
    smaller.setup_runtime()
    assert smaller._comm_buffer is not original._comm_buffer
    assert factory.call_count == 2
    assert [
        call.kwargs["num_max_tokens_per_rank"] for call in factory.call_args_list
    ] == [8192, 4096]
    original.setup_runtime()
    assert factory.call_count == 2
    wrapper.buffer.group.rank = lambda: 1
    with pytest.raises(ValueError, match="aligned EP group and expert ranks"):
        candidate.SM90FlashInferStrategy(cfg).setup_runtime()
    assert factory.call_count == 2


@pytest.mark.parametrize("graph", [False, True])
def test_ep1_bypasses_v2_and_preserves_local_contract(monkeypatch, graph):
    cfg = dataclasses.replace(
        _cfg(),
        ep_size=1,
        n_local_experts=384,
        local_expert_end=384,
        max_tokens_per_rank=8192,
        is_decode_role=graph,
    )
    _, factory, buf, _, _ = _install_v2(monkeypatch, cfg, graph=graph)
    strategy = candidate.SM90FlashInferStrategy(cfg)
    strategy.setup_runtime()
    result = strategy(*_inputs())
    assert strategy._comm_buffer is None
    factory.assert_not_called()
    buf.dispatch.assert_not_called()
    buf.combine.assert_not_called()
    assert result.dtype is torch.float32
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)
    if graph:
        assert strategy._local.prepare_calls == [(1, 6, True), (1, 6, False)]
        assert strategy._local.forward_calls[-1][4] is None
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="rows within"):
            strategy(*_inputs(2))
    else:
        assert not strategy._local.prepare_calls
        assert strategy._local.eager_calls[-1][3:] == (0, 384)


@pytest.mark.parametrize("comm", ["legacy", "ag"])
def test_stale_transport_overrides_are_rejected(monkeypatch, comm):
    cfg = _cfg()
    _install_v2(monkeypatch, cfg, graph=False)
    monkeypatch.setenv("DSV41_SM90_MOE_COMM", comm)
    with pytest.raises(ValueError, match="remove legacy/AG override"):
        candidate.SM90FlashInferStrategy(cfg)


def test_prefill_ignores_shared_decode_graph_flag(monkeypatch):
    cfg = dataclasses.replace(_cfg(), is_decode_role=False, max_tokens_per_rank=8192)
    _, factory, buf, _, _ = _install_v2(monkeypatch, cfg, graph=True)
    strategy = candidate.SM90FlashInferStrategy(cfg)
    assert strategy._decode_graph is False
    assert strategy._max_tokens_per_rank == 8192
    strategy.setup_runtime()
    assert factory.call_args.kwargs["num_max_tokens_per_rank"] == 8192
    assert not strategy._local.prepare_calls
    strategy(*_inputs())
    assert buf.dispatch.call_args.kwargs["do_cpu_sync"] is True
    assert strategy._local.eager_calls
    assert not strategy._local.forward_calls


@pytest.mark.parametrize(
    "override,recv_rows,expected_lengths",
    [
        (None, 32768, [32768]),
        (None, 32769, [32768, 1]),
        (None, 65539, [32768, 32768, 3]),
        (2, 3, [2, 1]),
    ],
)
def test_prefill_recv_chunk_keeps_one_collective_pair_and_bf16_rows(
    monkeypatch, override, recv_rows, expected_lengths
):
    # Exercise the real default threshold with tiny CPU hidden width, not a
    # model allocation. This is wiring/assembly coverage, not FI numerical proof.
    assert candidate._PREFILL_LOCAL_CHUNK_ROWS == 32768
    if override is not None:
        monkeypatch.setattr(candidate, "_PREFILL_LOCAL_CHUNK_ROWS", override)
    cfg = dataclasses.replace(
        _cfg(), dim=2, max_tokens_per_rank=65536, is_decode_role=False
    )
    _, _, buf, received, _ = _install_v2(
        monkeypatch, cfg, graph=False, recv_rows=recv_rows
    )
    strategy = candidate.SM90FlashInferStrategy(cfg)
    strategy.setup_runtime()
    rx, ri, rw, handle = received
    rx.copy_((torch.arange(recv_rows * 2).reshape(recv_rows, 2) % 251).bfloat16())
    ri[:, 0].copy_(torch.arange(recv_rows) % 48)
    rw[:, 0].copy_((torch.arange(recv_rows) % 127).float() / 128)
    original = [t.clone() for t in (rx, ri, rw)]
    calls = []

    def local(x, weights, ids, local_start, local_end):
        calls.append((x.clone(), weights.clone(), ids.clone()))
        assert (local_start, local_end) == (0, 48)
        assert x.is_contiguous() and weights.is_contiguous() and ids.is_contiguous()
        assert ids.dtype is torch.int64
        return x.float() + weights[:, :1] * 0.125 + ids[:, :1].float() / 128

    monkeypatch.setattr(strategy._local, "_forward_into_buf", local)
    # Echo the whole local receive output so every assembled row is observable.
    buf.combine.side_effect = lambda y, h: (y.clone(), None, None)
    result = strategy(
        torch.zeros((1, 2), dtype=torch.bfloat16),
        torch.ones((1, 6), dtype=torch.float32) / 6,
        torch.arange(6, dtype=torch.int64).reshape(1, 6),
    )
    assert [x.shape[0] for x, _, _ in calls] == expected_lengths
    offset = 0
    for x, weights, ids in calls:
        end = offset + x.shape[0]
        for actual, expected in (
            (x, rx[offset:end]),
            (weights, rw[offset:end]),
            (ids, ri[offset:end]),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        offset = end
    assert offset == recv_rows
    buf.dispatch.assert_called_once()
    buf.combine.assert_called_once()
    assert buf.combine.call_args.args[1] is handle
    combined_input = buf.combine.call_args.args[0]
    assert combined_input.dtype is torch.bfloat16
    expected = (rx.float() + rw[:, :1] * 0.125 + ri[:, :1].float() / 128).bfloat16()
    torch.testing.assert_close(combined_input, expected, rtol=0, atol=0)
    torch.testing.assert_close(result, expected.float(), rtol=0, atol=0)
    for actual, saved in zip((rx, ri, rw), original):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)


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
