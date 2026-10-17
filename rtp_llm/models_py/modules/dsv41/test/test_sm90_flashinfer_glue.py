"""Bit-exact independent Torch gates for standalone SM90 FI glue fusion.

No FlashInfer kernels or model accuracy thresholds are replaced by this target.
The reference copies actual tensor bits and masks invalid slots by selection,
never by multiplying poison by zero. Graph changes all four live inputs.
GPU/Bazel execution belongs to the parent agent.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def sm90_device():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (9, 0)


def _api():
    from rtp_llm.models_py.modules.dsv41.moe._sm90_flashinfer_glue import (
        finish_flashinfer_output,
        prepare_flashinfer_inputs,
    )

    return prepare_flashinfer_inputs, finish_flashinfer_output


def _bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    # Explicit canonical copies also work for singleton/empty strided tensors.
    aa = torch.empty(actual.numel(), dtype=actual.dtype, device=actual.device)
    bb = torch.empty(expected.numel(), dtype=expected.dtype, device=expected.device)
    aa.copy_(actual.reshape(-1))
    bb.copy_(expected.reshape(-1))
    torch.testing.assert_close(
        aa.view(torch.uint8), bb.view(torch.uint8), rtol=0, atol=0
    )


def _bit_pattern(shape, *, bf16):
    # Include both signs of zero, infinities, quiet/signalling NaNs with
    # noncanonical payload, and finite extremes. Do not construct via float().
    raw = (
        [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC1, 0xFFC5, 0x7F81, 0x0001]
        if bf16
        else [
            0x00000000,
            0x80000000,
            0x3F000000,
            0xBF000000,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0xFFC05678,
            0x7F801234,
            0x00000001,
        ]
    )
    width = 16 if bf16 else 32
    signed = [v if v < 2 ** (width - 1) else v - 2**width for v in raw]
    storage = torch.tensor(
        signed, device="cuda", dtype=torch.int16 if bf16 else torch.int32
    )
    count = 1
    for dim in shape:
        count *= dim
    return (
        storage.repeat((count + len(raw) - 1) // len(raw))[:count]
        .reshape(shape)
        .view(torch.bfloat16 if bf16 else torch.float32)
        .contiguous()
    )


def _inputs(rows, topk, dim, phase=0):
    x = _bit_pattern((rows, dim), bf16=True)
    weights = _bit_pattern((rows, topk), bf16=False)
    out = _bit_pattern((rows, dim), bf16=True).flip(-1).contiguous()
    extremes = torch.tensor(
        [0, 47, -1, 48, 2**32, 2**32 + 47, -(2**63), 2**63 - 1],
        device="cuda",
        dtype=torch.int64,
    )
    ids = extremes.repeat((rows * topk + 7) // 8)[: rows * topk].reshape(rows, topk)
    if phase == 1:
        ids.fill_(-1)
    elif phase == 2:
        ids = (
            torch.arange(rows * topk, device="cuda", dtype=torch.int64).reshape(
                rows, topk
            )
            % 48
        )
        x = x.flip(-1).contiguous()
        weights = weights.flip(-1).contiguous()
        out = out.flip(-1).contiguous()
    elif phase == 3 and rows:
        ids.fill_(-1)
        ids[::2, 0] = 47
        weights.fill_(-0.0)  # Zero route weight does NOT mean invalid row.
        out = out.roll(3, dims=-1).contiguous()
    return x, weights, ids.contiguous(), out


def _reference(x, weights, indices, out, experts, dtype):
    # Legality must be established in INT64 before INT32 conversion.
    valid = (indices >= 0) & (indices < experts)
    live = valid.any(dim=-1, keepdim=True)
    safe_x = torch.where(live, x, torch.zeros_like(x))
    safe_w = torch.where(valid, weights, torch.zeros_like(weights))
    safe_i = torch.where(valid, indices, experts).to(torch.int32)
    final = torch.where(live, out, torch.zeros_like(out)).to(dtype)
    return safe_x, safe_w, safe_i, live, final


@pytest.mark.parametrize(
    "rows,topk,dim", [(0, 6, 127), (1, 1, 1), (1, 6, 127), (8, 8, 5120), (32, 6, 129)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("phase", [0, 1, 2, 3])
@torch.no_grad()
def test_prepare_finish_independent_bitexact(rows, topk, dim, dtype, phase):
    prepare, finish = _api()
    x, weights, indices, out = _inputs(rows, topk, dim, phase)
    original = [t.clone() for t in (x, weights, indices, out)]
    expected = _reference(x, weights, indices, out, 48, dtype)
    prepared = prepare(x, weights, indices, 48)
    actual = (*prepared, finish(out, prepared[3], output_dtype=dtype))
    assert len(prepared) == 4
    for value, reference in zip(actual, expected):
        _bits(value, reference)
        assert value.is_contiguous()
        if value.numel():
            assert all(
                value.data_ptr() != source.data_ptr()
                for source in (x, weights, indices, out)
            )
    for value, saved in zip((x, weights, indices, out), original):
        _bits(value, saved)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.no_grad()
def test_live_graph_changes_all_inputs_and_masks(dtype):
    prepare, finish = _api()
    live = _inputs(8, 8, 129)

    def call():
        prepared = prepare(*live[:3], 48)
        return (*prepared, finish(live[3], prepared[3], output_dtype=dtype))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for phase in (1, 2, 3, 0):
        changed = _inputs(8, 8, 129, phase)
        for destination, source in zip(live, changed):
            destination.copy_(source)
        expected = _reference(*live, 48, dtype)
        eager = call()
        graph.replay()
        torch.cuda.synchronize()
        for value, ref, direct in zip(captured, expected, eager):
            _bits(value, ref)
            _bits(value, direct)
        for value, saved in zip(live, changed):
            _bits(value, saved)


@torch.no_grad()
def test_int64_range_and_max_expert_sentinel():
    prepare, finish = _api()
    maximum = 2**31 - 1
    x = _bit_pattern((1, 11), bf16=True)
    weights = _bit_pattern((1, 8), bf16=False)
    ids = torch.tensor(
        [
            [
                0,
                maximum - 1,
                maximum,
                2**32,
                2**32 + maximum - 1,
                -(2**63),
                2**63 - 1,
                -1,
            ]
        ],
        device="cuda",
        dtype=torch.int64,
    )
    expected = _reference(x, weights, ids, x, maximum, torch.float32)
    prepared = prepare(x, weights, ids, maximum)
    actual = (*prepared, finish(x, prepared[3]))
    for value, ref in zip(actual, expected):
        _bits(value, ref)


@torch.no_grad()
def test_explicit_layout_dtype_shape_and_expert_contract():
    prepare, finish = _api()
    x, weights, ids, out = _inputs(8, 8, 129)
    for experts in (0, -1, True, 1.5, 2**31):
        with pytest.raises((TypeError, ValueError)):
            prepare(x, weights, ids, experts)
    for bad_x, bad_weights, bad_ids in (
        (x.float(), weights, ids),
        (x, weights.bfloat16(), ids),
        (x, weights, ids.int()),
        (x.cpu(), weights, ids),
        (x[:, ::2], weights, ids),
        (x, weights[:, ::2], ids[:, ::2]),
        (x, weights.T, ids.T),
        (x, weights[:1], ids[:1]),
        (x, weights[:, :0], ids[:, :0]),
        (x[:, :0], weights, ids),
        (
            x,
            torch.zeros((8, 9), device="cuda"),
            torch.zeros((8, 9), dtype=torch.int64, device="cuda"),
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            prepare(bad_x, bad_weights, bad_ids, 48)
    live = torch.ones((8, 1), dtype=torch.bool, device="cuda")
    for bad_out, bad_live, dtype in (
        (out.float(), live, torch.float32),
        (out, live.float(), torch.float32),
        (out, live[:, 0], torch.float32),
        (out[:, ::2], live, torch.float32),
        (out, live.cpu(), torch.float32),
        (out, live, torch.float16),
        (out, live[:1], torch.float32),
    ):
        with pytest.raises((TypeError, ValueError)):
            finish(bad_out, bad_live, output_dtype=dtype)


@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
@torch.no_grad()
def test_recv_count_masks_poisoned_tail_on_changing_graph_replay(count_dtype):
    prepare, finish = _api()
    x, weights, ids, out = _inputs(8, 8, 129, phase=2)
    # Unused receive slots may contain valid IDs and NaNs from previous calls.
    x[1:].view(torch.int16).fill_(0x7FC5)
    weights[1:].view(torch.int32).fill_(0x7FC05678)
    ids[1:].fill_(47)
    original = [t.clone() for t in (x, weights, ids, out)]
    count = torch.tensor(1, device="cuda", dtype=count_dtype)

    def call():
        prepared = prepare(x, weights, ids, 48, recv_count=count)
        return (*prepared, finish(out, prepared[3]))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for rows in (0, 1, 8, 3, 0, 7, 8, 1):
        count.fill_(rows)
        active = torch.arange(8, device="cuda")[:, None] < count
        masked_ids = torch.where(active, ids, -1)
        expected = _reference(x, weights, masked_ids, out, 48, torch.float32)
        eager = call()
        graph.replay()
        torch.cuda.synchronize()
        for actual, direct, ref in zip(captured, eager, expected):
            _bits(actual, ref)
            _bits(actual, direct)
        for actual, saved in zip((x, weights, ids, out), original):
            _bits(actual, saved)


@torch.no_grad()
def test_recv_count_requires_one_device_integer():
    prepare, _ = _api()
    x, weights, ids, _ = _inputs(8, 8, 129)
    for count in (
        torch.tensor(1, dtype=torch.int32),
        torch.tensor(1.0, device="cuda"),
        torch.ones(2, device="cuda", dtype=torch.int64),
    ):
        with pytest.raises(ValueError):
            prepare(x, weights, ids, 48, recv_count=count)
