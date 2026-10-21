"""Bounded host registration lifecycle and real cross-range Graph reads."""

import mmap
from unittest.mock import patch

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.engram import _CudaArrayView
from rtp_llm.models_py.modules.dsv41.sm90_engram import _register_shared_host_ranges


class FakeRuntime:
    def __init__(self, fail_at=None, bad_mapping_at=None):
        self.calls = []
        self.released = []
        self.fail_at = fail_at
        self.bad_mapping_at = bad_mapping_at

    def register(self, runtime, address, size):
        self.calls.append((address, size))
        return (1 if len(self.calls) == self.fail_at else 0,)

    def cudaHostGetDevicePointer(self, address, flags):
        delta = 1 if len(self.calls) == self.bad_mapping_at else 0
        return 0, address + 0x10000000 + delta

    def cudaHostUnregister(self, address):
        self.released.append(address)
        return (0,)

    def cudaGetErrorName(self, result):
        return 0, b"injected_error"


def _register(runtime, chunk=8192):
    with patch(
        "rtp_llm.models_py.modules.dsv41.sm90_engram._register_shared_host_memory",
        side_effect=runtime.register,
    ):
        return _register_shared_host_ranges(runtime, 0x100000, 16384 + 37, chunk)


def test_ranges_cover_tail_without_overlap_and_keep_every_registration():
    runtime = FakeRuntime()
    addresses = _register(runtime)
    assert addresses == (0x100000, 0x102000, 0x104000)
    assert runtime.calls == [
        (addresses[0], 8192),
        (addresses[1], 8192),
        (addresses[2], 37),
    ]
    assert runtime.released == []
    whole = FakeRuntime()
    assert _register(whole, chunk=0) == (0x100000,)
    assert whole.calls == [(0x100000, 16384 + 37)]


def test_failed_register_rolls_back_successful_ranges_only_in_reverse():
    runtime = FakeRuntime(fail_at=3)
    with pytest.raises(RuntimeError, match="offset=16384"):
        _register(runtime)
    assert runtime.released == [0x102000, 0x100000]


def test_discontiguous_device_view_rejected_and_rolled_back():
    runtime = FakeRuntime(bad_mapping_at=2)
    with pytest.raises(RuntimeError, match="noncontiguous"):
        _register(runtime)
    assert runtime.released == [0x102000, 0x100000]
    with pytest.raises(ValueError, match="page multiple"):
        _register(FakeRuntime(), chunk=17)


@torch.inference_mode()
def test_real_graph_reads_across_registered_boundaries():
    from cuda.bindings import runtime

    assert torch.cuda.get_device_capability() == (9, 0)
    chunk = 64 * 1024
    size = 3 * chunk + 32
    mapping = mmap.mmap(-1, size)
    storage = torch.frombuffer(mapping, dtype=torch.uint8)
    storage.copy_(torch.arange(size, dtype=torch.int64).remainder(251).to(torch.uint8))
    addresses = ()
    view = owner = graph = None
    try:
        addresses = _register_shared_host_ranges(
            runtime, storage.data_ptr(), size, chunk
        )
        assert len(addresses) == 4
        result, device_pointer = runtime.cudaHostGetDevicePointer(storage.data_ptr(), 0)
        assert int(result) == 0
        owner = _CudaArrayView(storage, storage, device_pointer)
        view = torch.as_tensor(owner, device="cuda")
        indices_cpu = torch.tensor(
            [0, chunk - 1, chunk, 2 * chunk - 1, 2 * chunk, size - 1]
        )
        indices = indices_cpu.cuda()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                view.index_select(0, indices)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = view.index_select(0, indices)
        for step in range(3):
            current = (indices_cpu + step).remainder(size)
            indices.copy_(current)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output.cpu(), storage[current], rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph = view = owner = None
        for address in reversed(addresses):
            assert int(runtime.cudaHostUnregister(address)[0]) == 0
        storage = None
        mapping.close()
