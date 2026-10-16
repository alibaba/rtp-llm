"""Strict independent metadata oracle for the bounded vLLM alignment port.

No Humming/source override or production changes. Exact integer comparisons
include padding and inactive tail, not just the live expert blocks. The oracle
sorts Python route tuples and does not reproduce CUDA's histogram/prefix code.
GPU execution belongs to the dedicated Bazel target managed by the main agent.
"""

import pytest
import torch

from rtp_llm.models_py.kernels.cuda.sm90_moe_align import (
    UPSTREAM_REVISION,
    UPSTREAM_SOURCE_SHA256,
    align_small_batch,
)


def _oracle(cpu_ids, experts, block_size, local_start=0):
    assert cpu_ids.device.type == "cpu"
    values = cpu_ids.flatten().tolist()
    slots = len(values)
    stride = max(experts, 32)
    capacity_bound = min(slots * block_size, slots + experts * (block_size - 1))
    capacity = ((capacity_bound + block_size - 1) // block_size) * block_size
    # The upstream CTA orders each expert's slots by counting-thread, then its
    # strided input iteration. This is not global slot order and not an oracle
    # for route reduction order; consumers must retain original slot semantics.
    routes = sorted(
        (
            (value - local_start, slot % stride, slot // stride, slot)
            for slot, value in enumerate(values)
            if local_start <= value < local_start + experts
        )
    )
    sorted_slots = []
    block_experts = []
    for expert in range(experts):
        group = [entry[3] for entry in routes if entry[0] == expert]
        blocks = (len(group) + block_size - 1) // block_size
        sorted_slots.extend(group + [slots] * (blocks * block_size - len(group)))
        block_experts.extend([expert] * blocks)
    padded = len(sorted_slots)
    sorted_slots.extend([slots] * (capacity - padded))
    block_experts.extend([-1] * (capacity // block_size - len(block_experts)))
    return tuple(
        torch.tensor(x, dtype=torch.int32)
        for x in (sorted_slots, block_experts, [padded])
    )


def _check(cpu_ids, actual, experts, block_size, local_start=0):
    assert actual is not None
    expected = _oracle(cpu_ids, experts, block_size, local_start)
    assert len(actual) == 3
    for result, reference in zip(actual, expected):
        assert result.is_cuda and result.dtype == torch.int32
        assert result.is_contiguous()
        torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)
    # Separate membership check: every valid slot occurs exactly once, no
    # invalid slot survives. This doesn't depend on the oracle sort key.
    live_slots = actual[0].cpu()
    live_slots = live_slots[live_slots < cpu_ids.numel()].tolist()
    wanted = [
        i for i, value in enumerate(cpu_ids.flatten().tolist())
        if local_start <= value < local_start + experts
    ]
    assert sorted(live_slots) == wanted


@pytest.fixture(scope="module", autouse=True)
def _sm90_required():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Dedicated SM90 alignment target requires SM90; skip is not PASS")
    print(f"VLLM_ALIGN_PIN revision={UPSTREAM_REVISION} source_sha256={UPSTREAM_SOURCE_SHA256}")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("experts", [1, 31, 32, 48, 64])
@pytest.mark.parametrize("block_size", [1, 8, 32, 128, 256])
def test_exact_irregular_and_repeated(dtype, experts, block_size):
    # 17*8: several CTA-stride iterations, repeated expert IDs and padded top8.
    generator = torch.Generator().manual_seed(4281 + experts)
    cpu = torch.randint(-3, experts + 3, (17, 8), generator=generator, dtype=dtype)
    cpu[:, 6:] = -1
    cpu[:4, :6] = 0
    _check(cpu, align_small_batch(cpu.cuda(), experts, block_size), experts, block_size)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(0, 8), (1, 8), (127, 8), (341, 3)])
def test_empty_all_invalid_and_capacity_tail(dtype, shape):
    # 1023 slots is the maximum supported case, deliberately not warp aligned.
    for value in (-1, 48):
        cpu = torch.full(shape, value, dtype=dtype)
        _check(cpu, align_small_batch(cpu.cuda(), 48, 64), 48, 64)
    if cpu.numel():
        cpu.flatten()[::3] = 47
        cpu.flatten()[1::5] = 0
        _check(cpu, align_small_batch(cpu.cuda(), 48, 64), 48, 64)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("local_start", [0, 48, 336, 2**31 - 1 - 48])
def test_global_range_filter_and_wide_invalid(dtype, local_start):
    values = [
        -1, local_start - 1, local_start, local_start + 47,
        local_start + 48, 0, local_start + 7, local_start + 7,
    ]
    if dtype == torch.int64:
        # In particular, do not narrow 2**32 + a valid ID back to that ID.
        values += [-(2**63), 2**63 - 1, 2**32 + local_start, -(2**32) + local_start]
        values += [-1] * 4
    cpu = torch.tensor(values, dtype=dtype).reshape(-1, 8)
    _check(
        cpu, align_small_batch(cpu.cuda(), 48, 16, local_start=local_start),
        48, 16, local_start,
    )


def test_current_stream_and_graph_route_changes():
    # Compile before capture, then repeatedly change the routing values without
    # changing shape/capacity. Captured metadata must not retain stale counts,
    # slots, expert tails or invalid IDs. Includes non-default stream capture.
    start, experts, block_size = 96, 48, 32
    cpu = torch.full((17, 8), -1, dtype=torch.int64)
    device = cpu.cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            warm = align_small_batch(device, experts, block_size, local_start=start)
    stream.synchronize()
    _check(cpu, warm, experts, block_size, start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = align_small_batch(device, experts, block_size, local_start=start)
    pointers = tuple(t.data_ptr() for t in captured)
    cases = [
        torch.full_like(cpu, start + 47),
        torch.full_like(cpu, -1),
        torch.arange(cpu.numel(), dtype=cpu.dtype).reshape_as(cpu) % 53 + start - 2,
        torch.full_like(cpu, start),
        torch.full_like(cpu, 2**32 + start),
    ]
    for changed in cases:
        with torch.cuda.stream(stream):
            device.copy_(changed)
            graph.replay()
        stream.synchronize()
        assert tuple(t.data_ptr() for t in captured) == pointers
        _check(changed, captured, experts, block_size, start)


def test_unsupported_metadata_gate():
    ids = torch.zeros((1, 8), dtype=torch.int64, device="cuda")
    assert align_small_batch(ids.cpu(), 48, 32) is None
    assert align_small_batch(ids.float(), 48, 32) is None
    assert align_small_batch(ids.flatten(), 48, 32) is None
    assert align_small_batch(ids[:, ::2], 48, 32) is None
    assert align_small_batch(torch.empty((1, 0), device="cuda", dtype=torch.int64), 48, 32) is None
    assert align_small_batch(torch.empty((128, 8), device="cuda", dtype=torch.int64), 48, 32) is None
    for experts, block, start in [(0, 32, 0), (65, 32, 0), (48, 0, 0), (48, 257, 0), (48, 32, -1)]:
        assert align_small_batch(ids, experts, block, local_start=start) is None
