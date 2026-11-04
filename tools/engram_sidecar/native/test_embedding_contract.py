#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only external Engram embedding contract regression against real source.

Run in a torch + numpy environment, without an extension build or GPU access:
  timeout --kill-after=5s 60s python test_embedding_contract.py \
      --repo /path/to/RTP-LLM

Real torch supplies dtype/device objects; table tensors, storage and Triton
lookup are explicit test doubles. This verifies Python admission, identity,
dispatch and lifetime contracts, NOT native EGM, kernel correctness or CUDA
Graph safety. It does not replace the GPU integration tests.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

TORCH = None
ENGRAM = None


def contiguous_strides(shape):
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


class FakeTensor:
    """Metadata only: no backing bytes and no tensor allocation or conversion."""

    def __init__(self, shape, *, dtype=None, device="cuda:0", strides=None):
        self.shape = tuple(shape)
        self.dtype = TORCH.uint8 if dtype is None else dtype
        self.device = TORCH.device(device)
        self._strides = contiguous_strides(self.shape) if strides is None else tuple(strides)
        self.copy_attempts = []

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def stride(self):
        return self._strides

    def is_contiguous(self):
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self._strides)):
            if size != 1 and stride != expected:
                return False
            expected *= size
        return True

    def _forbidden(self, operation, *args, **kwargs):
        self.copy_attempts.append(operation)
        raise AssertionError("external table must not use " + operation)

    def clone(self, *args, **kwargs):
        return self._forbidden("clone", *args, **kwargs)

    def to(self, *args, **kwargs):
        return self._forbidden("to", *args, **kwargs)

    def contiguous(self, *args, **kwargs):
        return self._forbidden("contiguous", *args, **kwargs)

    def copy_(self, *args, **kwargs):
        return self._forbidden("copy_", *args, **kwargs)

    def cpu(self, *args, **kwargs):
        return self._forbidden("cpu", *args, **kwargs)

    def cuda(self, *args, **kwargs):
        return self._forbidden("cuda", *args, **kwargs)

    def numpy(self, *args, **kwargs):
        return self._forbidden("numpy", *args, **kwargs)

    def view(self, *args, **kwargs):
        return self._forbidden("view", *args, **kwargs)

    def detach(self, *args, **kwargs):
        return self._forbidden("detach", *args, **kwargs)

    def __array__(self, *args, **kwargs):
        return self._forbidden("__array__", *args, **kwargs)


class FakeStorage:
    """Retains the two original views and counts admission / close activity."""

    def __init__(self, weight=None, scales=None, *, device=0, head_dim=64):
        self.weight = FakeTensor((3, head_dim)) if weight is None else weight
        self.scales = FakeTensor((3, head_dim // 32)) if scales is None else scales
        self.device = device
        self._lock = threading.RLock()
        self.closed = False
        self.active_uses = 0
        self.use_entries = self.use_exits = self.close_calls = 0
        self.requested_layers = []

    @contextmanager
    def use(self):
        with self._lock:
            if self.closed:
                raise RuntimeError("storage is not READY")
            self.active_uses += 1
            self.use_entries += 1
            try:
                yield
            finally:
                self.active_uses -= 1
                self.use_exits += 1

    def table_views(self, layer_id):
        if self.active_uses <= 0:
            raise AssertionError("table_views must be called within storage.use()")
        self.requested_layers.append(layer_id)
        return self.weight, self.scales

    def close(self):
        self.closed = True
        self.close_calls += 1


@contextmanager
def forbid_allocations():
    """Catch torch factory copies in addition to FakeTensor method copies."""
    operations = ("tensor", "as_tensor", "from_numpy", "frombuffer", "clone",
                  "empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like")
    with ExitStack() as stack:
        mocks = []
        for operation in operations:
            mocks.append(stack.enter_context(patch.object(
                TORCH, operation,
                side_effect=AssertionError("unexpected torch allocation/copy: " + operation))))
        yield
        for mock in mocks:
            mock.assert_not_called()


@contextmanager
def fake_triton(lookup):
    """Do not import RTP package initializers or real Triton on a CPU runner."""
    names = ("rtp_llm", "rtp_llm.models_py", "rtp_llm.models_py.modules",
             "rtp_llm.models_py.modules.dsv41")
    modules = {}
    previous = None
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        if previous is not None:
            setattr(previous, name.rsplit(".", 1)[1], module)
        previous = module
    triton = ModuleType(names[-1] + "._engram_triton")
    triton.lookup_host_rows = lookup
    previous._engram_triton = triton
    modules[triton.__name__] = triton
    # Start on another device so a missing guard cannot accidentally pass.
    device_state = SimpleNamespace(current_device=1, events=[])

    @contextmanager
    def device_context(device):
        if isinstance(device, int):
            target = device
        else:
            device = TORCH.device(device)
            if device.type != "cuda" or device.index is None:
                raise AssertionError("device guard requires an explicit CUDA ordinal")
            target = device.index
        original = device_state.current_device
        device_state.events.append(("enter", original, target))
        device_state.current_device = target
        try:
            yield
        finally:
            device_state.events.append(("exit", device_state.current_device, original))
            device_state.current_device = original

    with patch.dict(sys.modules, modules), \
            patch.object(TORCH.cuda, "device", side_effect=device_context), \
            patch.object(TORCH.cuda, "current_device", side_effect=lambda: device_state.current_device):
        yield device_state


class EmbeddingContractTest(unittest.TestCase):
    def assert_no_copies(self, storage):
        self.assertEqual(storage.weight.copy_attempts, [])
        self.assertEqual(storage.scales.copy_attempts, [])

    def rejected(self, storage, message):
        with patch.object(TORCH.cuda, "get_device_properties") as properties:
            with forbid_allocations(), self.assertRaisesRegex(ValueError, message):
                ENGRAM.HostEngramEmbedding.from_storage(storage, 7)
            properties.assert_not_called()
        self.assertEqual(storage.active_uses, 0)
        self.assertEqual(storage.use_entries, storage.use_exits)
        self.assertEqual(storage.close_calls, 0)
        self.assert_no_copies(storage)

    def create(self, storage):
        with patch.object(TORCH.cuda, "get_device_properties",
                          return_value=SimpleNamespace(multi_processor_count=80)) as properties:
            with forbid_allocations():
                result = ENGRAM.HostEngramEmbedding.from_storage(storage, 7)
            properties.assert_called_once_with(storage.weight.device)
        self.assert_no_copies(storage)
        return result

    def test_rejects_non_cuda_views(self):
        for which in ("weight", "scales", "both"):
            with self.subTest(which=which):
                weight = FakeTensor((3, 64), device="cpu" if which != "scales" else "cuda:0")
                scales = FakeTensor((3, 2), device="cpu" if which != "weight" else "cuda:0")
                self.rejected(FakeStorage(weight, scales), "storage CUDA device")

    def test_rejects_mismatched_or_unspecified_device(self):
        cases = (
            FakeStorage(scales=FakeTensor((3, 2), device="cuda:1")),
            FakeStorage(device=1),
            FakeStorage(FakeTensor((3, 64), device="cuda"), FakeTensor((3, 2), device="cuda")),
        )
        for storage in cases:
            with self.subTest(weight=str(storage.weight.device), scales=str(storage.scales.device),
                              storage_device=storage.device):
                self.rejected(storage, "storage CUDA device")

    def test_rejects_non_uint8_weight_or_scale(self):
        for which in ("weight", "scales"):
            with self.subTest(which=which):
                weight = FakeTensor((3, 64), dtype=TORCH.float32 if which == "weight" else TORCH.uint8)
                scales = FakeTensor((3, 2), dtype=TORCH.float32 if which == "scales" else TORCH.uint8)
                self.rejected(FakeStorage(weight, scales), "preserve FP8/E8M0 as uint8")

    def test_rejects_invalid_weight_matrix_shape(self):
        for shape in ((3,), (3, 64, 1), (0, 64)):
            with self.subTest(shape=shape):
                self.rejected(FakeStorage(weight=FakeTensor(shape)), "nonempty matrix")

    def test_rejects_mismatched_scale_geometry(self):
        for shape in ((4, 2), (3, 3), (3, 2, 1), (6,)):
            with self.subTest(shape=shape):
                self.rejected(FakeStorage(scales=FakeTensor(shape)), "scale shape")

    def test_rejects_non_power_of_two_or_small_head_dim(self):
        for dim in (0, 16, 48, 96, 160):
            with self.subTest(head_dim=dim):
                self.rejected(FakeStorage(head_dim=dim), "power-of-two width >=32")

    def test_rejects_non_contiguous_weight_or_scale_strides(self):
        cases = (
            FakeStorage(weight=FakeTensor((3, 64), strides=(65, 1))),
            FakeStorage(weight=FakeTensor((3, 64), strides=(128, 2))),
            FakeStorage(scales=FakeTensor((3, 2), strides=(3, 1))),
            FakeStorage(scales=FakeTensor((3, 2), strides=(4, 2))),
        )
        for storage in cases:
            with self.subTest(weight_stride=storage.weight.stride(), scale_stride=storage.scales.stride()):
                self.rejected(storage, "must be contiguous")

    def test_success_keeps_exact_external_views_without_private_tables(self):
        for dim in (32, 64, 128):
            with self.subTest(head_dim=dim):
                storage = FakeStorage(head_dim=dim)
                embedding = self.create(storage)
                try:
                    self.assertIsNone(embedding.weight)
                    self.assertIsNone(embedding.scales)
                    self.assertIsNone(embedding._pinned)
                    self.assertIs(embedding._external_storage, storage)
                    self.assertIs(embedding._uva[0], storage.weight)
                    self.assertIs(embedding._uva[1], storage.scales)
                    self.assertEqual(embedding.block_size, 32)
                    self.assertEqual(embedding._num_sms, 80)
                    self.assertTrue(embedding._uses_external_storage)
                    self.assertFalse(embedding._external_closed)
                    self.assertEqual(storage.requested_layers, [7])
                    self.assertEqual((storage.use_entries, storage.use_exits, storage.active_uses), (1, 1, 0))
                    self.assertEqual(storage.close_calls, 0)
                    self.assert_no_copies(storage)
                finally:
                    embedding.release_external_views()

    def test_lookup_dispatch_preserves_view_identity_under_storage_guard(self):
        storage = FakeStorage()
        embedding = self.create(storage)
        indices = FakeTensor((2, 1), dtype=TORCH.int64)
        output = object()

        def dispatch(weight, scales, got_indices, num_sms):
            self.assertGreater(storage.active_uses, 0)
            self.assertEqual(device_state.current_device, storage.device)
            self.assertEqual(TORCH.cuda.current_device(), storage.device)
            self.assertIs(weight, storage.weight)
            self.assertIs(scales, storage.scales)
            self.assertIs(got_indices, indices)
            self.assertEqual(num_sms, 80)
            return output

        lookup = Mock(side_effect=dispatch)
        try:
            with fake_triton(lookup) as device_state, forbid_allocations():
                self.assertEqual(device_state.current_device, 1)
                self.assertIs(embedding(indices, "cuda:0"), output)
                self.assertEqual(device_state.current_device, 1)
                self.assertEqual(device_state.events, [("enter", 1, 0), ("exit", 0, 1)])
            lookup.assert_called_once()
            self.assertEqual((storage.use_entries, storage.use_exits, storage.active_uses), (2, 2, 0))
            self.assertEqual(indices.copy_attempts, [])
            self.assert_no_copies(storage)
        finally:
            embedding.release_external_views()

    def test_lookup_exception_restores_original_cuda_device(self):
        storage = FakeStorage()
        embedding = self.create(storage)
        indices = FakeTensor((2, 1), dtype=TORCH.int64)

        def fail_lookup(*args):
            self.assertEqual(device_state.current_device, storage.device)
            raise RuntimeError("injected lookup failure")

        lookup = Mock(side_effect=fail_lookup)
        try:
            with fake_triton(lookup) as device_state, forbid_allocations():
                with self.assertRaisesRegex(RuntimeError, "injected lookup failure"):
                    embedding(indices, "cuda:0")
                self.assertEqual(device_state.current_device, 1)
                self.assertEqual(device_state.events, [("enter", 1, 0), ("exit", 0, 1)])
            lookup.assert_called_once()
            self.assertEqual(storage.active_uses, 0)
            self.assertEqual(storage.use_entries, storage.use_exits)
            self.assert_no_copies(storage)
        finally:
            embedding.release_external_views()

    def test_lookup_rejects_wrong_indices_or_target_device(self):
        storage = FakeStorage()
        embedding = self.create(storage)
        lookup = Mock(side_effect=AssertionError("invalid lookup reached Triton"))
        cases = ((FakeTensor((1, 1), dtype=TORCH.int64, device="cpu"), "cuda:0"),
                 (FakeTensor((1, 1), dtype=TORCH.int64, device="cuda:1"), "cuda:0"),
                 (FakeTensor((1, 1), dtype=TORCH.int64), "cpu"),
                 (FakeTensor((1, 1), dtype=TORCH.int64), "cuda:1"))
        try:
            with fake_triton(lookup), forbid_allocations():
                for indices, target in cases:
                    with self.subTest(indices=str(indices.device), target=target):
                        with self.assertRaisesRegex(ValueError, "indices on its CUDA device"):
                            embedding(indices, target)
            lookup.assert_not_called()
            self.assertEqual(storage.active_uses, 0)
            self.assert_no_copies(storage)
        finally:
            embedding.release_external_views()

    def test_lookup_rejects_invalid_index_dtype_or_rank(self):
        storage = FakeStorage()
        embedding = self.create(storage)
        lookup = Mock(side_effect=AssertionError("invalid indices reached Triton"))
        cases = (FakeTensor((2, 1), dtype=TORCH.uint8),
                 FakeTensor((2, 1), dtype=TORCH.float32),
                 FakeTensor((2,), dtype=TORCH.int64))
        try:
            with fake_triton(lookup), forbid_allocations():
                for indices in cases:
                    with self.subTest(dtype=str(indices.dtype), shape=indices.shape):
                        with self.assertRaisesRegex(
                                ValueError,
                                "external Engram lookup requires a 2D int32/int64 index tensor"):
                            embedding(indices, "cuda:0")
                        self.assertEqual(indices.copy_attempts, [])
            lookup.assert_not_called()
            self.assertEqual(storage.active_uses, 0)
            self.assert_no_copies(storage)
        finally:
            embedding.release_external_views()

    def test_release_is_idempotent_and_use_after_release_fails(self):
        storage = FakeStorage()
        embedding = self.create(storage)
        embedding.release_external_views()
        embedding.release_external_views()
        self.assertIsNone(embedding._uva)
        self.assertIsNone(embedding._external_storage)
        self.assertTrue(embedding._external_closed)
        self.assertTrue(embedding._uses_external_storage)
        self.assertIsNone(embedding.weight)
        self.assertIsNone(embedding.scales)
        self.assertEqual(storage.close_calls, 0)  # Caller still owns storage lifecycle.
        self.assertFalse(storage.closed)
        lookup = Mock(side_effect=AssertionError("closed embedding reached Triton"))
        with fake_triton(lookup), forbid_allocations():
            with self.assertRaisesRegex(RuntimeError, "external Engram embedding is closed"):
                embedding(FakeTensor((1, 1), dtype=TORCH.int64), "cuda:0")
        lookup.assert_not_called()
        self.assertEqual((storage.use_entries, storage.use_exits, storage.active_uses), (1, 1, 0))
        self.assert_no_copies(storage)

    def test_storage_admission_failure_does_not_query_cuda_or_copy(self):
        storage = FakeStorage()
        storage.close()
        with patch.object(TORCH.cuda, "get_device_properties") as properties:
            with forbid_allocations(), self.assertRaisesRegex(RuntimeError, "not READY"):
                ENGRAM.HostEngramEmbedding.from_storage(storage, 7)
            properties.assert_not_called()
        self.assertEqual(storage.requested_layers, [])
        self.assert_no_copies(storage)


def main():
    global TORCH, ENGRAM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    source = args.repo.resolve() / "rtp_llm/models_py/modules/dsv41/engram.py"
    if not source.is_file():
        parser.error("--repo must contain rtp_llm/models_py/modules/dsv41/engram.py")

    # This fresh process is intentionally CPU-only even on a shared GPU host.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    TORCH = torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU contract runner must start without initialized CUDA")
    spec = importlib.util.spec_from_file_location("_engram_embedding_contract_under_test", source)
    ENGRAM = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ENGRAM
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU test attempted CUDA initialization")):
        spec.loader.exec_module(ENGRAM)
        print(json.dumps(dict(stage="BEGIN", scope="CPU_CONTRACT_ONLY", torch=torch.__version__,
                              source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              gpu_validation=False)), flush=True)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(EmbeddingContractTest)
        result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    initialized = torch.cuda.is_initialized()
    passed = result.wasSuccessful() and not initialized
    print(json.dumps(dict(stage="END", status="PASS" if passed else "FAIL", scope="CPU_CONTRACT_ONLY",
                          tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                          cuda_initialized=initialized, gpu_validation=False)), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
