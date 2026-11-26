"""Executor numerics against unpacked FP4 weights and ordinary matmuls."""

import hashlib
import os
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_decode import (
    Sm120DecodeExperts,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_grouped_fp4 import (
    Sm120GroupedFp4Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.weight_adapter import (
    adapt_split_moe_weights,
)
from rtp_llm.resident_weights import (
    GLOBAL_SCOPE,
    ResidentWeights,
    WeightDict,
    resident_weights_of,
    store,
)
from rtp_llm.utils.model_weight import W


@pytest.mark.parametrize("capturing", [False, True])
def test_prequantized_input_rejected_before_capture_host_sync(capturing):
    executor = Sm120GroupedFp4Executor.__new__(Sm120GroupedFp4Executor)
    torch.nn.Module.__init__(executor)
    x = torch.zeros((1, 128), dtype=torch.float8_e4m3fn)
    scales = torch.ones((1, 4), dtype=torch.uint8)
    with (
        patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
        patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}),
        patch.object(executor, "forward_eager") as eager,
    ):
        with pytest.raises(ValueError, match="requires unquantized input"):
            executor.forward(x, torch.ones(1, 1), torch.zeros(1, 1), scales)
        eager.assert_not_called()


def _weights(experts, dim, inter, layout):
    weights, dense = {}, []
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    for w_key, s_key, out_dim, in_dim in (
        (W.moe_w1, W.moe_s1, 2 * inter, dim),
        (W.moe_w2, W.moe_s2, dim, inter),
    ):
        packed = torch.randint(
            0, 256, (experts, out_dim, in_dim // 2), dtype=torch.uint8, device="cuda"
        )
        scales = torch.randint(
            119, 124, (experts, out_dim, in_dim // 32), dtype=torch.uint8, device="cuda"
        )
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
        expanded_scales = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
        dense.append(table[codes] * expanded_scales)
        weights[w_key] = packed.view(torch.int8)
        weights[s_key] = scales.view(torch.float8_e8m0fnu)
    if layout == "up_gate":
        for key in (W.moe_w1, W.moe_s1):
            value = weights[key]
            weights[key] = torch.cat(
                (
                    value[:, inter:].view(torch.uint8),
                    value[:, :inter].view(torch.uint8),
                ),
                dim=1,
            ).view(value.dtype)
    return weights, dense


def _reference(x, weights, ids, dense, limit):
    w13, w2 = dense
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w13.size(0)):
        rows, slots = torch.where((ids == expert) & (weights != 0))
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w13[expert]).chunk(2, -1)
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        hidden = F.silu(gate) * up
        output = F.linear(hidden, w2[expert]) * weights[rows, slots, None]
        result.index_add_(0, rows, output)
    return result


@pytest.mark.parametrize("layout", ["gate_up", "up_gate"])
@pytest.mark.parametrize("rows", [1, 3, 129, 4097])
@pytest.mark.parametrize("backend", ["contiguous", "grouped"])
@pytest.mark.parametrize("limit", [0.0, 1.5])
def test_executor_matches_unpacked_reference(layout, rows, backend, limit):
    torch.manual_seed(1729)
    experts, dim, inter, topk = 16, 512, 512, 6
    cfg = SimpleNamespace(
        n_routed_experts=experts,
        n_activated_experts=topk,
        dim=dim,
        moe_inter_dim=inter,
        moe_w1_layout=layout,
        swiglu_limit=limit,
        max_tokens_per_rank=rows,
    )
    layer_weights, dense = _weights(experts, dim, inter, layout)
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device="cuda") * 0.2
    ids = torch.arange(topk, device="cuda").expand(rows, -1).clone()
    weights = torch.rand(rows, topk, device="cuda") / topk
    # Expert zero sees every token; ten experts see none. Masked slots must
    # contribute nothing, including when a valid expert id has zero weight.
    ids[:, -1] = -1
    weights[:, -2] = 0
    if rows > 1:
        weights[1] = 0
    expected = _reference(x, weights, ids, dense, cfg.swiglu_limit)
    if backend == "grouped":
        executor = Sm120GroupedFp4Executor.__new__(Sm120GroupedFp4Executor)
        torch.nn.Module.__init__(executor)
        executor.cfg = cfg
        executor.setup_weights(layer_weights)
        forward = executor.forward
    else:
        executor = Sm120DecodeExperts(cfg, layer_weights)
        forward = getattr(executor, "forward_" + backend)

    def check(actual):
        assert torch.isfinite(actual).all()
        # This is an unquantized reference; the executor rounds activations
        # to FP8 and GEMM outputs to BF16. Bound relative aggregate error.
        error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-10)
        assert error < 0.08, (backend, rows, layout, error.item())
        if rows > 1:
            torch.testing.assert_close(
                actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0
            )

    check(forward(x, weights, ids))
    if backend != "contiguous":
        with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
            forward(x, weights, ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward(x, weights, ids)
        captured.fill_(float("nan"))
        graph.replay()
        check(captured)


def test_grouped_fp4_supports_128_inter_dim():
    torch.manual_seed(17)
    experts, dim, inter, topk, rows = 8, 512, 128, 4, 5
    cfg = SimpleNamespace(
        n_routed_experts=experts,
        n_activated_experts=topk,
        dim=dim,
        moe_inter_dim=inter,
        moe_w1_layout="gate_up",
        swiglu_limit=0.0,
        max_tokens_per_rank=rows,
    )
    layer_weights, dense = _weights(experts, dim, inter, "gate_up")
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device="cuda") * 0.2
    ids = torch.arange(topk, device="cuda").expand(rows, -1).clone()
    weights = torch.rand(rows, topk, device="cuda") / topk
    expected = _reference(x, weights, ids, dense, cfg.swiglu_limit)
    executor = Sm120GroupedFp4Executor.__new__(Sm120GroupedFp4Executor)
    torch.nn.Module.__init__(executor)
    executor.cfg = cfg
    executor.setup_weights(layer_weights)
    actual = executor.forward_eager(x, weights, ids)
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-10)
    assert error < 0.08, error.item()


class _ModelWeights:
    """Weakref-able container matching the attributes ResidentWeights binds."""

    def __init__(self, layers, global_weights):
        self.weights = layers
        self.global_weights = global_weights


def _tensor_digest(tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1).clone()
    storage = flat.untyped_storage()
    raw = (
        torch.empty(0, dtype=torch.uint8, device=flat.device)
        .set_(storage, 0, (storage.nbytes(),), (1,))
        .cpu()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _private_resident_store():
    prev_store = os.environ.get("RTP_LLM_RESIDENT_STORE")
    os.environ["RTP_LLM_RESIDENT_STORE"] = (
        f"rtp_llm_sm120_resident_test.{uuid.uuid4().hex}"
    )
    store_name = store.default_store_name()
    store.ensure_running(store_name, owner_pid=os.getpid())
    try:
        yield store_name
    finally:
        try:
            store.stop_store(store_name)
        except ConnectionRefusedError:
            pass
        if prev_store is None:
            os.environ.pop("RTP_LLM_RESIDENT_STORE", None)
        else:
            os.environ["RTP_LLM_RESIDENT_STORE"] = prev_store


def _make_split_moe_weights(experts: int, dim: int, inter: int) -> dict:
    def packed(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            0, 256, (experts, out_dim, in_dim // 2), dtype=torch.uint8, device="cuda"
        ).view(torch.int8)

    def scale(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            119, 124, (experts, out_dim, in_dim // 32), dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e8m0fnu)

    return {
        "cp.router": torch.randn(experts, dim, dtype=torch.bfloat16, device="cuda"),
        "cp.routed_gate": packed(inter, dim),
        "cp.routed_up": packed(inter, dim),
        "cp.routed_gate_scale": scale(inter, dim),
        "cp.routed_up_scale": scale(inter, dim),
        "cp.routed_down": packed(dim, inter),
        "cp.routed_down_scale": scale(dim, inter),
    }


def _executor_weight_digests(executor) -> tuple[str, ...]:
    if isinstance(executor, Sm120GroupedFp4Executor):
        tensors = (executor._w13, executor._w2, executor._s13_sm120, executor._s2_sm120)
    else:
        tensors = (executor._w13, executor._w2, executor._s13, executor._s2)
    return tuple(_tensor_digest(t) for t in tensors)


def _instantiate_sm120_executor(executor_cls, cfg, layer_weights):
    if hasattr(executor_cls, "setup_weights"):
        executor = executor_cls.__new__(executor_cls)
        torch.nn.Module.__init__(executor)
        executor.cfg = cfg
        executor.setup_weights(layer_weights)
        return executor
    return executor_cls(cfg, layer_weights)


def _run_resident_round_trip(
    executor_cls,
    raw_weights: dict,
    *,
    fingerprint: dict,
    prepare_layer=None,
    assert_cold_layer=None,
):
    experts, dim, inter, topk = 16, 512, 512, 6
    cfg = SimpleNamespace(
        n_routed_experts=experts,
        n_activated_experts=topk,
        dim=dim,
        moe_inter_dim=inter,
        moe_w1_layout="gate_up",
        swiglu_limit=0.0,
        max_tokens_per_rank=8,
    )
    x = torch.randn(4, dim, dtype=torch.bfloat16, device="cuda") * 0.2
    ids = torch.arange(topk, device="cuda").expand(4, -1).contiguous()
    wts = torch.full((4, topk), 1.0 / topk, dtype=torch.float32, device="cuda")

    with _private_resident_store() as store_name:
        cold_mgr = ResidentWeights("target", 0, dict(fingerprint), False)
        assert cold_mgr.ensure_store_running()
        mw = _ModelWeights(
            [WeightDict(0, raw_weights)],
            WeightDict(GLOBAL_SCOPE, {}),
        )
        cold_mgr.move_to_vmm(mw)
        cold_layer = mw.weights[0]
        if assert_cold_layer is not None:
            assert_cold_layer(cold_layer)
        if prepare_layer is not None:
            prepare_layer(cold_layer, inter)
        cold_exec = _instantiate_sm120_executor(executor_cls, cfg, cold_layer)

        cold_out = cold_exec.forward(x, wts, ids).clone()
        assert cold_out.float().abs().mean().item() > 1e-3
        cold_digests = _executor_weight_digests(cold_exec)
        cold_mgr.finalize()
        assert (
            f"{torch.cuda.get_device_properties(0).uuid}/target"
            in store.list_store(store_name).slots
        )
        del cold_exec, cold_layer, mw

        hot_mgr = ResidentWeights("target", 0, dict(fingerprint), False)
        assert hot_mgr.ensure_store_running()
        restored = hot_mgr.attach()
        assert restored is not None
        hot_mw = _ModelWeights([{}], {})
        for item in restored:
            target = (
                hot_mw.global_weights
                if item.scope == GLOBAL_SCOPE
                else hot_mw.weights[item.scope]
            )
            target[item.name] = item.tensor
        hot_mgr.bind(hot_mw)
        hot_layer = hot_mw.weights[0]
        if prepare_layer is not None:
            prepare_layer(hot_layer, inter)
        assert W.moe_w1 not in hot_layer

        with patch.object(
            executor_cls,
            "_build_weights",
            side_effect=AssertionError("builder must not run on resident restart"),
        ):
            hot_exec = _instantiate_sm120_executor(executor_cls, cfg, hot_layer)
        hot_mgr.finalize()
        assert _executor_weight_digests(hot_exec) == cold_digests
        hot_out = hot_exec.forward(x, wts, ids)
        assert _tensor_digest(hot_out) == _tensor_digest(cold_out)


@pytest.mark.parametrize("executor_cls", [Sm120GroupedFp4Executor, Sm120DecodeExperts])
def test_sm120_executors_replay_resident_weights(executor_cls):
    torch.manual_seed(20261125)
    names = {
        "router": "cp.router",
        "routed_gate": "cp.routed_gate",
        "routed_up": "cp.routed_up",
        "routed_gate_scale": "cp.routed_gate_scale",
        "routed_up_scale": "cp.routed_up_scale",
        "routed_down": "cp.routed_down",
        "routed_down_scale": "cp.routed_down_scale",
    }
    _run_resident_round_trip(
        executor_cls,
        _make_split_moe_weights(16, 512, 512),
        fingerprint={"model_type": "deepseek_v4", "executor": executor_cls.__name__},
        prepare_layer=lambda layer, inter: adapt_split_moe_weights(
            layer, inter, 0, names
        ),
    )


def test_sm120_grouped_fp4_replays_direct_vmm_gate_up_weights():
    torch.manual_seed(20261126)
    layer_weights, _ = _weights(16, 512, 512, "gate_up")
    _run_resident_round_trip(
        Sm120GroupedFp4Executor,
        layer_weights,
        fingerprint={
            "model_type": "deepseek_v4",
            "executor": "Sm120GroupedFp4ExecutorDirectVmm",
        },
        assert_cold_layer=lambda layer: _assert_vmm_backed(layer[W.moe_w1]),
    )


def _assert_vmm_backed(tensor: torch.Tensor) -> None:
    assert resident_weights_of(tensor) is not None
