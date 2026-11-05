"""CPU contracts for copying external scale views into existing scale shards.

Tensor metadata and symmetric memory are doubles. These tests exercise the
production cache/admission/slice-copy flow without initializing CUDA or a group.
Real peer addressing and Fabric copies remain GPU integration responsibilities.
"""

from contextlib import ExitStack, contextmanager, nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


spec = importlib.util.spec_from_file_location(
    "_engram_external_scale_source_under_test",
    Path(__file__).resolve().parents[1] / "_engram_scale_shards.py",
)
provider = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = provider
spec.loader.exec_module(provider)


class ScaleTensor:
    """Small independent byte rows with CUDA metadata and observable copies."""

    def __init__(self, data, *, device="cuda:0", dtype=torch.uint8, shape=None, pointer=4096):
        self.data = data
        self.device = torch.device(device)
        self.dtype = dtype
        self.shape = (len(data), 8) if shape is None else shape
        self.ndim = len(self.shape)
        self.requires_grad = False
        self.pointer = pointer
        self.slices = []

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def is_contiguous(self): return True
    def is_conj(self): return False
    def is_neg(self): return False
    def data_ptr(self): return self.pointer
    def numel(self): return len(self.data) * 8

    def cpu(self):
        raise AssertionError("external scales must never stage a full CPU table")

    def to(self, *_args, **_kwargs):
        raise AssertionError("external singleton must copy rather than alias via to(device)")

    def clone(self):
        return ScaleTensor([row[:] for row in self.data], device=self.device)

    def fill_(self, value):
        for row in self.data:
            row[:] = [value] * 8
        return self

    def __getitem__(self, selected):
        self.slices.append(selected)
        # Keep references to destination rows so copy_ writes into the parent.
        return ScaleTensor(self.data[selected], device=self.device)

    def copy_(self, source):
        for target, row in zip(self.data, source.data):
            target[:] = row
        return self


class ExternalScaleSourceTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU regression initialized CUDA")))
        self.stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch.object(torch, "Tensor", ScaleTensor))
        self.stack.enter_context(patch.object(provider, "_GENERATIONS", {}))

    @contextmanager
    def collective(self, rank, size=4):
        with ExitStack() as stack:
            stack.enter_context(patch.object(provider.dist, "is_initialized", return_value=True))
            stack.enter_context(patch.object(provider.dist, "get_world_size", return_value=size))
            stack.enter_context(patch.object(provider.dist, "get_rank", return_value=rank))
            stack.enter_context(patch.object(provider, "_members", return_value=tuple(range(size))))
            stack.enter_context(patch.object(provider, "_local_metadata", return_value={"uuid": f"gpu-{rank}", "eligible": True}))
            stack.enter_context(patch.object(provider, "_eligible_members", return_value=True))
            stack.enter_context(patch.object(provider, "_can_access_peers", return_value=True))
            exchange = stack.enter_context(patch.object(
                provider, "_exchange", side_effect=lambda _group, payload, _phase: [dict(payload) for _ in range(size)]
            ))
            stack.enter_context(patch.object(torch.cuda, "device", return_value=nullcontext()))
            stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=Mock()))
            stack.enter_context(patch.object(torch, "tensor", side_effect=lambda data, **kw: ScaleTensor(data, shape=(len(data),), **kw)))
            symm = ModuleType("torch.distributed._symmetric_memory")
            symm.empty = Mock(side_effect=lambda shape, **kw: ScaleTensor(
                [[0] * 8 for _ in range(shape[0])], pointer=4096 + rank * 256, **kw
            ))
            symm.rendezvous = Mock(side_effect=lambda local, **_kw: SimpleNamespace(
                rank=rank, world_size=size, buffer_ptrs=[4096 + index * 256 for index in range(size)],
                buffer_size=local.numel(),
            ))
            stack.enter_context(patch.dict(sys.modules, {symm.__name__: symm}))
            stack.enter_context(patch.object(torch.distributed, "_symmetric_memory", symm, create=True))
            yield SimpleNamespace(group_name="external-scale-test"), symm, exchange

    def test_external_singleton_owns_copy_and_rejects_wrong_device(self):
        source = ScaleTensor([[index] * 8 for index in range(5)])
        for group in (None, object()):
            with self.subTest(group=group), \
                    patch.object(provider.dist, "is_initialized", return_value=True), \
                    patch.object(provider.dist, "get_world_size", return_value=1):
                copied = provider.create_sharded_scales(source, group, "cuda:0", None, external_source=True)
                self.assertIsNot(copied, source)
                self.assertEqual(copied.data, source.data)
                copied.data[0][0] = 99
                self.assertEqual(source.data[0][0], 0)
        with self.assertRaisesRegex(ValueError, "same-device CUDA"):
            provider.create_sharded_scales(source, None, "cuda:1", None, external_source=True)
        with self.assertRaisesRegex(ValueError, "CPU"):
            provider.create_sharded_scales(source, None, "cuda:0", None)

    def test_default_cpu_source_still_uses_existing_transfer(self):
        source = ScaleTensor([[42] * 8], device="cpu")
        transferred = object()
        with patch.object(source, "to", return_value=transferred) as transfer, \
                patch.object(source, "clone", side_effect=AssertionError("legacy CPU source cloned")):
            self.assertIs(provider.create_sharded_scales(source, None, "cuda:0", None), transferred)
            transfer.assert_called_once_with(torch.device("cuda:0"))

    def test_external_sources_require_explicit_same_device_and_geometry(self):
        for source in (
            ScaleTensor([[0] * 8], device="cpu"),
            ScaleTensor([[0] * 8], device="cuda:1"),
            ScaleTensor([[0] * 7], shape=(1, 7)),
            ScaleTensor([], shape=(0, 8)),
        ):
            with self.subTest(device=source.device, shape=source.shape), self.collective(0) as (group, symm, _):
                with self.assertRaisesRegex(RuntimeError, "metadata disagree"):
                    provider.create_sharded_scales(source, group, "cuda:0", "table", external_source=True)
                symm.empty.assert_not_called()

    def test_rank_copy_covers_rows_once_with_padding_and_cache_hits_read_nothing(self):
        # Non-divisible and empty-tail cases detect accidental full copies and
        # off-by-one slices independently of the production partition formula.
        for rows, expected in ((7, ((0, 2), (2, 4), (4, 6), (6, 7))),
                               (2, ((0, 1), (1, 2), (2, 2), (2, 2)))):
            for rank, (begin, end) in enumerate(expected):
                with self.subTest(rows=rows, rank=rank), self.collective(rank) as (group, symm, exchange):
                    source = ScaleTensor([[index] * 8 for index in range(rows)])
                    descriptor = provider.create_sharded_scales(
                        source, group, "cuda:0", f"rows-{rows}", external_source=True
                    )
                    capacity = 2 if rows == 7 else 1
                    wanted = source.data[begin:end] + [[127] * 8 for _ in range(capacity - (end - begin))]
                    self.assertEqual(descriptor.local_scales.data, wanted)
                    self.assertEqual(source.slices, [slice(begin, end)] if end > begin else [])
                    self.assertEqual(symm.empty.call_count, 1)
                    self.assertEqual(symm.rendezvous.call_count, 1)
                    source.slices.clear()
                    cached = provider.create_sharded_scales(
                        source, group, "cuda:0", f"rows-{rows}", external_source=True
                    )
                    self.assertIs(cached, descriptor)
                    self.assertEqual(source.slices, [])
                    self.assertEqual(symm.empty.call_count, 1)
                    self.assertEqual([call.args[2] for call in exchange.call_args_list],
                                     ["table_metadata", "peer_access", "allocation", "ready", "table_metadata", "peer_access"])


if __name__ == "__main__":
    unittest.main()
