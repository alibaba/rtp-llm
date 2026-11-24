"""The prefill position synthesis must not fall back to a pageable H2D.

``_build_positions_from_lengths`` receives the framework's pinned int32
``input_lengths`` / ``prefix_lengths`` and needs int64 on the device. A
dtype-converting ``.to(device=..., dtype=torch.int64)`` converts on the host
first, and that intermediate is unpinned, so the transfer becomes a blocking
pageable copy. The profile of an online prefill showed exactly that. These
tests pin the contract so the regression cannot come back.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.modules.dsv4.prefill.forward import (
    _int64_on_device as dsv4_int64_on_device,
)
from rtp_llm.models_py.modules.dsv41.prefill.forward import (
    _int64_on_device as dsv41_int64_on_device,
)


def _pinned_int32(values):
    return torch.tensor(values, dtype=torch.int32).pin_memory()


def _memcpy_kinds(fn, repeats=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    return [event.key for event in prof.key_averages() if "Memcpy" in event.key]


def test_upload_is_pinned_not_pageable():
    source = _pinned_int32(list(range(12)))
    assert source.is_pinned()

    kinds = _memcpy_kinds(lambda: dsv41_int64_on_device(source, torch.device("cuda")))
    assert any("Pinned -> Device" in kind for kind in kinds), kinds
    assert not any("Pageable -> Device" in kind for kind in kinds), kinds

    # The conversion the previous code performed is the regression this guards.
    legacy = _memcpy_kinds(
        lambda: source.to(device=torch.device("cuda"), dtype=torch.int64)
    )
    assert any("Pageable -> Device" in kind for kind in legacy), legacy


def test_dsv41_matches_the_dtype_converting_to():
    device = torch.device("cuda")
    for size in (1, 9, 10, 12, 13, 32):
        for dtype in (torch.int32, torch.int64, torch.bool):
            source = _pinned_int32([index % 5 for index in range(size)]).to(dtype)
            expected = source.to(device=device, dtype=torch.int64)
            actual = dsv41_int64_on_device(source, device)
            assert actual.dtype == torch.int64
            assert actual.device.type == "cuda"
            assert actual.shape == (size,)
            assert actual.is_contiguous()
            assert torch.equal(actual, expected), (size, dtype)


def test_dsv4_helper_has_the_same_contract():
    device = torch.device("cuda")
    source = _pinned_int32([3, 1, 4, 1, 5])
    assert torch.equal(
        dsv4_int64_on_device(source, device),
        source.to(device=device, dtype=torch.int64),
    )


def test_cuda_source_is_a_plain_cast():
    device = torch.device("cuda")
    source = torch.arange(6, dtype=torch.int32, device=device)
    actual = dsv41_int64_on_device(source, device)
    assert actual.dtype == torch.int64
    assert actual.device.type == "cuda"
    assert torch.equal(actual, source.to(torch.int64))


def test_positions_match_the_reference_construction():
    device = torch.device("cuda")
    input_lengths = _pinned_int32([2, 3, 1])
    prefix_lengths = _pinned_int32([10, 20, 30])

    got = _build_positions_for_test(input_lengths, prefix_lengths, device)
    expected = torch.tensor([10, 11, 20, 21, 22, 30], dtype=torch.int64, device=device)
    assert torch.equal(got, expected), (got.tolist(), expected.tolist())


def _build_positions_for_test(input_lengths, prefix_lengths, device):
    from rtp_llm.models_py.modules.dsv41.prefill.forward import (
        _build_positions_from_lengths,
    )

    return _build_positions_from_lengths(
        input_lengths, prefix_lengths, device, total_tokens=6
    )
