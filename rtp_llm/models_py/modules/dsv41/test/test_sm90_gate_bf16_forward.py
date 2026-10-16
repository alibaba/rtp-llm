"""Real Gate.forward contract for the proposed SM90 small-M cast elision.

Requires the production hook to be enabled before positive tests can pass.
Uses real BF16/FP32 linear projections and the real dsv41 fused kernel; spies
only observe dtype/dispatch. The SM100 guard test mocks a host branch, not GPU
execution capability, and is not SM100 numerical evidence. Parent owns GPU.
"""

import pytest
import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4 import _record_tensor
from rtp_llm.models_py.modules.dsv41.moe import gate as gate_module
from rtp_llm.utils.model_weight import W


@pytest.fixture(autouse=True)
def _sm90_only(monkeypatch):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (
        9,
        0,
    ), "Requires a parent-locked SM90 GPU; missing hardware is not PASS"
    monkeypatch.setenv("DSV4_GATE_FP32", "0")
    monkeypatch.setenv("DSV4_GATE_FUSED", "1")
    monkeypatch.setattr(_record_tensor, "ENABLED", False)


def _make_case(rows=1, dim=128, *, hashed=False, visual=False, device="cuda"):
    torch.manual_seed(24927)
    experts, topk, vocab = 384, 6, 32
    weight = torch.randn(experts, dim, device=device, dtype=torch.bfloat16) * 0.125
    layer_weights = {
        W.v4_router_w: weight,
        W.v4_router_bias: torch.randn(experts, device=device) * 0.1,
    }
    if hashed:
        layer_weights[W.v4_router_tid2eid] = (
            torch.arange(vocab, device=device)[:, None] * 7
            + torch.arange(topk, device=device)[None]
        ) % experts
    if visual:
        layer_weights[W.v4_router_bias_vl] = torch.randn(experts, device=device) * 0.1
    gate = gate_module.Gate(
        layer_id=0,
        dim=dim,
        n_routed_experts=experts,
        n_activated_experts=topk,
        n_hash_layers=int(hashed),
        vocab_size=vocab,
        route_scale=2.5,
        layer_weights=layer_weights,
    )
    gate._has_visual_tokens = visual
    x = torch.randn(rows, dim, device=device, dtype=torch.bfloat16)
    ids = torch.arange(rows, device=device, dtype=torch.int64) % vocab
    if visual:
        ids[::2] += vocab
    return gate, x, ids


def _old_reference(gate, x, ids, *, force_fp32=False, fused=True):
    """Original materialized-FP32 pipeline; finite fallback cases use eager ops.

    No prospective eligibility helper is reused. Fused correctness itself has
    an independent eager oracle in test_sm90_gate_bf16_input.py.
    """
    if not x.shape[0]:
        return x.new_empty((0, gate.topk), dtype=torch.float32), ids.new_empty(
            (0, gate.topk)
        )
    if force_fp32:
        scores = F.linear(x.float(), gate.weight.float())
    else:
        scores = F.linear(x.bfloat16(), gate.weight.bfloat16()).float()
    visual = gate._has_visual_tokens
    if fused and not gate.hash and not visual and gate.score_func == "sqrtsoftplus":
        return gate_module.fused_sqrtsoftplus_gate(
            scores.contiguous(),
            gate.bias.contiguous(),
            topk=gate.topk,
            route_scale=2.5,
            norm_eps=1e-12,
        )
    if gate.score_func == "softmax":
        original = scores.softmax(-1)
    elif gate.score_func == "sigmoid":
        original = scores.sigmoid()
    else:
        original = F.softplus(scores).sqrt()
    ranking = original
    image_mask = ids >= gate.vocab_size if visual else None
    if gate.bias is not None:
        bias = (
            torch.where(image_mask[:, None], gate.bias_vl, gate.bias)
            if visual
            else gate.bias
        )
        ranking = original + bias
    if gate.hash:
        indices = gate.tid2eid[
            torch.where(image_mask, 0, ids) if visual else ids
        ].long()
        if visual:
            vision = (original + gate.bias_vl).topk(gate.topk, dim=-1).indices
            indices = torch.where(image_mask[:, None], vision.long(), indices)
    else:
        indices = ranking.topk(gate.topk, dim=-1).indices
    weights = original.gather(1, indices)
    if gate.score_func != "softmax":
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-12)
    return weights * 2.5, indices


def _bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a = torch.empty(actual.numel(), device=actual.device, dtype=actual.dtype)
    b = torch.empty(expected.numel(), device=expected.device, dtype=expected.dtype)
    a.copy_(actual.reshape(-1))
    b.copy_(expected.reshape(-1))
    torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


def _outputs_exact(actual, expected):
    assert actual[0].dtype == torch.float32 and actual[1].dtype == torch.int64
    for a, e in zip(actual, expected):
        _bits(a, e)


def _observe_kernel(monkeypatch):
    observed = []
    original = gate_module.fused_sqrtsoftplus_gate

    def record(scores, bias, **kwargs):
        observed.append((scores.dtype, bias.dtype))
        return original(scores, bias, **kwargs)

    monkeypatch.setattr(gate_module, "fused_sqrtsoftplus_gate", record)
    return observed


@pytest.mark.parametrize(
    "rows,dim", [(1, 5120), (2, 128), (4, 128), (8, 128), (32, 128)]
)
@torch.no_grad()
def test_real_forward_sm90_small_m_uses_bf16_kernel_input(monkeypatch, rows, dim):
    gate, x, ids = _make_case(rows, dim)
    saved = x.clone(), gate.weight.clone(), gate.bias.clone(), ids.clone()
    expected = _old_reference(gate, x, ids)
    observed = _observe_kernel(monkeypatch)
    actual = gate(x, ids)
    assert observed == [(torch.bfloat16, torch.float32)]
    _outputs_exact(actual, expected)
    for value, snapshot in zip((x, gate.weight, gate.bias, ids), saved):
        _bits(value, snapshot)


@pytest.mark.parametrize(
    "mode",
    [
        "fp32",
        "debug",
        "hash",
        "visual",
        "hash_visual",
        "fusedoff",
        "softmax",
        "sigmoid",
        "large_m",
        "grad",
        "otherarch",
        "cpu",
        "empty",
    ],
)
def test_real_forward_ineligible_paths_keep_original_contract(monkeypatch, mode):
    rows = 33 if mode == "large_m" else 0 if mode == "empty" else 4
    gate, x, ids = _make_case(
        rows,
        hashed=mode in ("hash", "hash_visual"),
        visual=mode in ("visual", "hash_visual"),
        device="cpu" if mode == "cpu" else "cuda",
    )
    force_fp32 = mode == "fp32"
    fused = mode not in ("fusedoff", "cpu")
    if force_fp32:
        monkeypatch.setenv("DSV4_GATE_FP32", "1")
    if not fused:
        monkeypatch.setenv("DSV4_GATE_FUSED", "0")
    if mode in ("softmax", "sigmoid"):
        gate.score_func = mode
    records = []
    if mode == "debug":
        monkeypatch.setattr(_record_tensor, "ENABLED", True)
        gate._dbg_prefix = "test_gate"
        monkeypatch.setattr(
            _record_tensor,
            "record_if_level",
            lambda level, name, tensor: records.append((name, tensor.dtype)),
        )
    with torch.no_grad():
        expected = _old_reference(gate, x, ids, force_fp32=force_fp32, fused=fused)
    observed = _observe_kernel(monkeypatch)
    if mode == "otherarch":
        # Branch-only evidence on actual SM90, not a claim to execute SM100.
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0)
        )
    if mode == "cpu":

        def no_cuda_query(*args, **kwargs):
            raise AssertionError("CPU fallback must not query CUDA capability")

        monkeypatch.setattr(torch.cuda, "get_device_capability", no_cuda_query)
    with torch.set_grad_enabled(mode == "grad"):
        actual = gate(x, ids)
    _outputs_exact(actual, expected)
    wants_fused = (
        fused
        and not gate.hash
        and not gate._has_visual_tokens
        and gate.score_func == "sqrtsoftplus"
        and rows > 0
    )
    assert observed == ([(torch.float32, torch.float32)] if wants_fused else [])
    if mode == "debug":
        assert ("test_gate_linear_scores", torch.float32) in records


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


@torch.no_grad()
def test_real_forward_graph_changes_input_router_weight_and_bias(monkeypatch):
    gate, x, ids = _make_case(4)
    old_graph, old_output = _capture(lambda: _old_reference(gate, x, ids))
    observed = _observe_kernel(monkeypatch)
    graph, output = _capture(lambda: gate(x, ids))
    assert observed and all(dtype == torch.bfloat16 for dtype, _ in observed)
    for step in range(4):
        torch.manual_seed(24100 + step)
        x.copy_(torch.randn_like(x))
        gate.weight.copy_(torch.randn_like(gate.weight) * 0.125)
        gate.bias.copy_(torch.randn_like(gate.bias) * 0.1)
        ids.copy_((ids + 1) % gate.vocab_size)
        if step == 2:
            gate.weight.zero_()
            gate.bias.zero_()
        saved = x.clone(), gate.weight.clone(), gate.bias.clone(), ids.clone()
        graph.replay()
        old_graph.replay()
        _outputs_exact(output, old_output)
        _outputs_exact(output, _old_reference(gate, x, ids))
        for value, snapshot in zip((x, gate.weight, gate.bias, ids), saved):
            _bits(value, snapshot)
