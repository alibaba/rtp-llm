"""Bitwise Prefill producer equivalence to the existing norm/pack chain.

Run explicitly on a coordinated CUDA device; no model/RTP ops are imported.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


def helper():
    path = Path(__file__).resolve().parents[3] / "triton_kernels/minimax_m31_gemma_rope.py"
    spec = importlib.util.spec_from_file_location("m31_norm_prefill_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.minimax_m31_gemma_norm_rope_


def same_bytes(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("rows", [0, 1, 7, 8, 9, 2049])
def test_prefill_outputs_match_legacy_and_preserve_projection(rows):
    norm = helper()
    torch.manual_seed(100901)
    # Match fused-projection views: all three inputs share one strided carrier.
    carrier = torch.randn(rows, 9856, device="cuda", dtype=torch.bfloat16)
    qkv, iq, ik = carrier[:, :9216], carrier[:, 9216:9728], carrier[:, 9728:]
    weights = tuple(torch.randn(128, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    angles = torch.randn(257, 32, device="cuda", dtype=torch.float32)
    cache = torch.cat((angles.cos(), angles.sin()), dim=1)
    positions = torch.arange(rows, device="cuda", dtype=torch.int32) % 257
    outputs = (
        torch.empty(rows, 64, 128, device="cuda", dtype=torch.bfloat16),
        torch.empty(rows, 4, 128, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(rows, 1152, device="cuda", dtype=torch.bfloat16),
    )
    kwargs = dict(num_q_heads=64, num_kv_heads=4, num_index_heads=4)
    for scale in (0, 0.001, 1, 1024):
        # Reuse fixed output storage with new values/positions, including tails
        # that do not fill GROUP=8, to catch stale output state.
        carrier.normal_().mul_(scale)
        positions.copy_(torch.randint(0, 257, (rows,), device="cuda", dtype=torch.int32))
        original = carrier.clone()
        legacy = carrier.clone()
        norm(legacy[:, :9216], legacy[:, 9216:9728], legacy[:, 9728:],
             weights, positions, cache, **kwargs)
        norm(qkv, iq, ik, weights, positions, cache, prefill_outputs=outputs, **kwargs)
        same_bytes(carrier, original)
        same_bytes(outputs[0], legacy[:, :8192].reshape(rows, 64, 128))
        same_bytes(outputs[1], legacy[:, 9216:9728].reshape(rows, 4, 128).to(torch.float8_e4m3fn))
        expected_pack = torch.cat((legacy[:, 8192:8704], legacy[:, 8704:9216], legacy[:, 9728:]), dim=1)
        same_bytes(outputs[2], expected_pack)


def test_prefill_rejects_projection_alias_and_legacy_output_mix():
    norm = helper()
    rows = 1
    carrier = torch.zeros(rows, 9856, device="cuda", dtype=torch.bfloat16)
    weights = (torch.ones(128, device="cuda", dtype=torch.bfloat16),) * 4
    positions = torch.zeros(rows, device="cuda", dtype=torch.int32)
    cache = torch.zeros(1, 64, device="cuda", dtype=torch.float32)
    iq8 = torch.empty(rows, 4, 128, device="cuda", dtype=torch.float8_e4m3fn)
    packed = torch.empty(rows, 1152, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(num_q_heads=64, num_kv_heads=4, num_index_heads=4)
    args = (carrier[:, :9216], carrier[:, 9216:9728], carrier[:, 9728:], weights, positions, cache)
    with pytest.raises(ValueError, match="independent non-input storage"):
        norm(*args, prefill_outputs=(carrier[:, :8192].reshape(rows, 64, 128), iq8, packed), **kwargs)
    q = torch.empty(rows, 64, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cannot be combined"):
        norm(*args, prefill_outputs=(q, iq8, packed), query_fp8_outputs=(iq8, iq8), **kwargs)


def test_independent_constant_head_oracle_high_position_graph_replay():
    # Constant positive heads normalize to 1 after BF16 rounding at eps=1e-6.
    # gamma=1.5 and exact quarter-turn cache values give an analytic oracle
    # independent of the legacy kernel's reduction implementation.
    norm = helper()
    rows, high = 9, 81921
    carrier = torch.ones(rows, 9856, device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros(rows, device="cuda", dtype=torch.int32)
    weights = tuple(torch.full((128,), 0.5, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    cache = torch.zeros(high + 1, 64, device="cuda", dtype=torch.float32)
    cache[:, :32] = 1
    cache[high, :32] = 0
    cache[high, 32:] = 1
    outputs = (
        torch.empty(rows, 64, 128, device="cuda", dtype=torch.bfloat16),
        torch.empty(rows, 4, 128, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(rows, 1152, device="cuda", dtype=torch.bfloat16),
    )
    kwargs = dict(num_q_heads=64, num_kv_heads=4, num_index_heads=4, prefill_outputs=outputs)

    def produce():
        norm(carrier[:, :9216], carrier[:, 9216:9728], carrier[:, 9728:],
             weights, positions, cache, **kwargs)

    produce()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        produce()
    for value, position in ((1, high), (8, 0), (2, high)):
        carrier.fill_(value)
        # V has a separate, unnormalized value and must survive every replay.
        carrier[:, 8704:9216] = value + 3
        positions.fill_(position)
        original = carrier.clone()
        graph.replay()
        expected = torch.full((128,), 1.5, device="cuda", dtype=torch.bfloat16)
        if position == high:
            expected[:32] = -1.5
        same_bytes(outputs[0], expected.expand(rows, 64, 128))
        same_bytes(outputs[1], expected.to(torch.float8_e4m3fn).expand(rows, 4, 128))
        packed = outputs[2]
        same_bytes(packed[:, :512].reshape(rows, 4, 128), expected.expand(rows, 4, 128))
        same_bytes(packed[:, 1024:], expected.expand(rows, 128))
        same_bytes(packed[:, 512:1024], original[:, 8704:9216])
        same_bytes(carrier, original)
