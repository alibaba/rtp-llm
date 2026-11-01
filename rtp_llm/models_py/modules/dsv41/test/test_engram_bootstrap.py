# SPDX-License-Identifier: Apache-2.0
"""File/bootstrap CPU tests; native mapping is mocked, never model forward.

Hash tests instantiate the real EngramLayout/NgramHashState with an explicit
token map. They need CPU PyTorch, but no CUDA device. File checks run without it.
Actual GPU reads and cross-host imports belong to the integration harness.
"""
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "engram_bootstrap_cpu_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
b = importlib.import_module(PACKAGE + ".engram_bootstrap")
m = importlib.import_module(PACKAGE + ".engram_manifest")
fixture_spec = importlib.util.spec_from_file_location(
    "engram_bootstrap_manifest_fixture", ROOT / "test" / "test_engram_manifest.py"
)
fixture_module = importlib.util.module_from_spec(fixture_spec)
sys.modules[fixture_spec.name] = fixture_module
fixture_spec.loader.exec_module(fixture_module)
try:
    import torch
except ImportError:
    torch = None


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def real_hash_state():
    e = importlib.import_module(PACKAGE + ".engram")
    config = dict(
        engram_layer_ids=[1, 14], engram_num_embeddings=[128, 256],
        engram_max_ngram_size=3, engram_n_heads=2, engram_head_dim=32,
        engram_compressed_vocab_size=8, engram_pad_token_id=2, engram_vocab_size=5,
    )
    return e.NgramHashState(e.EngramLayout(config), token_map=list(range(8)))


class BundleFiles:
    def prepare_bundle(self, layout=None, hash_abi=None, shards=1):
        self.temporary = tempfile.TemporaryDirectory(prefix="engram-bootstrap-ut-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "launch.json"
        self.layout = layout or types.SimpleNamespace(
            layer_ids=(1, 14), num_embeddings=(3, 5), head_dim=32
        )
        self.abi = hash_abi or "sha256:" + "b" * 64
        self.document = fixture_module.fixture()
        rows_by_layer = dict(zip(self.layout.layer_ids, self.layout.num_embeddings))
        for tensor in self.document["tensors"]:
            lid = int(tensor["name"].split(".")[1])
            rows = rows_by_layer[lid]
            columns = self.layout.head_dim if tensor["name"].endswith(".weight") else self.layout.head_dim // 32
            tensor["shape"] = [rows, columns]
            part = tensor["shards"][0]
            part.update(row_end=rows, logical_bytes=rows * columns,
                        byte_offset=0 if tensor["name"].endswith(".weight") else rows * self.layout.head_dim,
                        allocation_bytes=((rows * (self.layout.head_dim + self.layout.head_dim // 32) + 4095) // 4096) * 4096)
            tensor["row_bytes"] = columns
            if shards > 1:
                from uuid import uuid4, UUID
                step = (rows + shards - 1) // shards
                tensor["shards"] = [dict(part, shard_id=str(uuid4()), allocation_id=str(uuid4()),
                    owner_id=f"host-{i}", owner_incarnation=str(UUID(int=10 + i)),
                    row_begin=i * step, row_end=min((i + 1) * step, rows),
                    byte_offset=0, allocation_granularity=1, allocation_bytes=step * columns,
                    logical_bytes=(min((i + 1) * step, rows) - i * step) * columns)
                    for i in range(shards)]
        self.document["hash_abi"] = self.abi
        self.document["dataset_id"] = m.compute_dataset_id(self.document)
        self.manifest_raw = json.dumps(self.document).encode()
        self.samples = {
            tensor["name"]: {
                "row_ids": sorted({0, tensor["shape"][0] // 2, tensor["shape"][0] - 1} |
                                  {row for part in tensor["shards"] for row in (part["row_begin"], part["row_end"] - 1)}),
                "sha256": "sha256:" + "c" * 64,
            }
            for tensor in self.document["tensors"]
        }
        self.package = dict(
            raw=self.manifest_raw,
            dataset_id=self.document["dataset_id"], generation=self.document["generation"],
            model_revision=self.document["model_revision"],
            capabilities={part["allocation_id"]: b"private-handle-bytes"
                          for tensor in self.document["tensors"] for part in tensor["shards"]},
            samples=self.samples,
        )


class HashSemanticTest(unittest.TestCase):
    def test_real_hash_state_is_deterministic(self):
        self.assertEqual(b.hash_state_digest(real_hash_state()), b.hash_state_digest(real_hash_state()))

    def test_every_hash_constant_changes_digest(self):
        baseline = b.hash_state_digest(real_hash_state())
        for name in ("token_map", "multipliers", "primes", "offsets"):
            with self.subTest(constant=name):
                state = real_hash_state()
                tensor = state.layout.offsets if name == "offsets" else getattr(state, name)
                tensor.reshape(-1)[-1] += 1
                self.assertNotEqual(b.hash_state_digest(state), baseline)

    def test_pad_and_layout_semantics_change_digest(self):
        baseline = b.hash_state_digest(real_hash_state())
        for name, value in (("pad_id", 3), ("pad_token_id", 3), ("max_ngram_size", 4),
                            ("head_dim", 64), ("n_heads", 3),
                            ("compressed_vocab_size", 9), ("layer_ids", (1, 15)),
                            ("num_embeddings", (129, 256))):
            with self.subTest(field=name):
                state = real_hash_state()
                setattr(state if name == "pad_id" else state.layout, name, value)
                self.assertNotEqual(b.hash_state_digest(state), baseline)

    def test_equal_noncontiguous_constants_have_equal_digest(self):
        state = real_hash_state()
        baseline = b.hash_state_digest(state)
        state.token_map = state.token_map.repeat_interleave(2)[::2]
        self.assertFalse(state.token_map.is_contiguous())
        self.assertEqual(b.hash_state_digest(state), baseline)

    def test_non_cpu_or_wrong_dtype_rejected_without_gpu_allocation(self):
        for device, dtype in (("meta", torch.int64), ("cpu", torch.int32), ("cpu", torch.float32)):
            state = real_hash_state()
            state.token_map = torch.empty(state.token_map.shape, dtype=dtype, device=device)
            with self.assertRaisesRegex(b.BootstrapError, "HASH_CONSTANTS_MUST_BE_CPU_INT64"):
                b.hash_state_digest(state)


@unittest.skipIf(torch is None, "CPU PyTorch required; native/GPU calls are mocked")
class AttachFailureTest(BundleFiles, unittest.TestCase):
    def setUp(self):
        self.hash_state = real_hash_state()
        self.prepare_bundle(self.hash_state.layout, b.hash_state_digest(self.hash_state), shards=2)
        self.storage = mock.Mock()
        self.storage.verify.return_value = None
        self.storage.close.return_value = None
        self.native = types.SimpleNamespace(max_allocation_bytes=128 << 30)
        self.retained = []
        self.native_patch = mock.patch.object(b, "_load_native", return_value=self.native)
        self.storage_patch = mock.patch.object(b.FabricEngramStorage, "attach", return_value=self.storage)
        self.retain_patch = mock.patch.object(b, "retain_for_process", side_effect=self.retained.append)
        self.native_mock = self.native_patch.start()
        self.attach_mock = self.storage_patch.start()
        self.retain_patch.start()
        self.addCleanup(self.native_patch.stop)
        self.addCleanup(self.storage_patch.stop)
        self.addCleanup(self.retain_patch.stop)

    def do_attach(self, device="cuda:2", environ=None):
        return b.attach_from_package(self.package, layout=self.layout,
                                    hash_state=self.hash_state, device=device,
                                    environ={} if environ is None else environ)

    def test_explicit_p_mismatch_rejected_before_native_load(self):
        with self.assertRaisesRegex(ValueError, "COUNT_MISMATCH"):
            self.do_attach(environ={"ENGRAM_SHARD_COUNT": "3"})
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_single_owner_deployment_is_removed(self):
        self.prepare_bundle(self.hash_state.layout, b.hash_state_digest(self.hash_state))
        with self.assertRaisesRegex(ValueError, "2_TO_64"):
            self.do_attach()
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_native_capacity_mismatch_rejects_before_first_import(self):
        for maximum in (None, True, 0, 1):
            self.native.max_allocation_bytes = maximum
            with self.assertRaisesRegex(b.BootstrapError, "NATIVE_ALLOCATION_LIMIT"):
                self.do_attach()
            self.attach_mock.assert_not_called()

    def test_explicit_policy_is_forwarded_as_one_immutable_limits_object(self):
        env = {"RTP_ENGRAM_MAX_ALLOCATION_BYTES": str(128 << 30),
               "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": str(256 << 30)}
        self.do_attach(environ=env)
        limits = self.attach_mock.call_args.kwargs["limits"]
        self.assertEqual(limits.max_allocation_bytes, 128 << 30)
        self.assertEqual(limits.max_total_allocation_bytes, 256 << 30)

    def test_bad_policy_fails_before_native_load(self):
        with self.assertRaisesRegex(b.BootstrapError, "INVALID_CAPACITY_LIMIT"):
            self.do_attach(environ={"RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": "0"})
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_success_calls_verify_before_returning_storage(self):
        self.assertIs(self.do_attach(), self.storage)
        self.attach_mock.assert_called_once()
        self.assertEqual(self.attach_mock.call_args.kwargs["device"], 2)
        self.assertIs(self.attach_mock.call_args.kwargs["native"], self.native)
        self.storage.verify.assert_called_once()
        self.storage.close.assert_not_called()
        self.assertEqual(self.retained, [])  # Model-level success owns retention.

    def test_cpu_device_rejected_before_read_or_import(self):
        with self.assertRaisesRegex(b.BootstrapError, "CUDA_DEVICE_REQUIRED"):
            self.do_attach("cpu")
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_hash_mismatch_rejected_before_native_import(self):
        self.hash_state.token_map[-1] -= 1
        with self.assertRaisesRegex(b.BootstrapError, "HASH_ABI_MISMATCH"):
            self.do_attach()
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_missing_sample_rejected_before_native_import(self):
        self.package["samples"].pop(next(iter(self.package["samples"])))
        with self.assertRaisesRegex(b.BootstrapError, "INCOMPLETE_READ_SAMPLES"):
            self.do_attach()
        self.attach_mock.assert_not_called()

    def test_import_cleanup_pending_resource_is_retained(self):
        pending = b.CleanupPendingError("injected import cleanup failure", self.storage)
        self.attach_mock.side_effect = pending
        with self.assertRaises(b.CleanupPendingError) as raised:
            self.do_attach()
        self.assertIs(raised.exception, pending)
        self.assertEqual(self.retained, [self.storage])
        self.storage.verify.assert_not_called()

    def test_verify_failure_closes_storage_and_preserves_original_error(self):
        failure = b.BootstrapError("GPU_READ_SAMPLE_MISMATCH")
        self.storage.verify.side_effect = failure
        with self.assertRaises(b.BootstrapError) as raised:
            self.do_attach()
        self.assertIs(raised.exception, failure)
        self.storage.close.assert_called_once()
        self.assertEqual(self.retained, [])

    def test_verify_and_close_failure_retains_resource(self):
        failure = RuntimeError("injected asynchronous verification failure")
        self.storage.verify.side_effect = failure
        self.storage.close.side_effect = RuntimeError("injected drain failure")
        with self.assertRaises(b.CleanupPendingError) as raised:
            self.do_attach()
        self.assertIs(raised.exception.resource, self.storage)
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(self.retained, [self.storage])

    def test_sample_callback_compares_bytes_and_synchronizes(self):
        validated = m.validate_for_attach(m.parse_manifest(self.manifest_raw),
                    m.ExpectedLayout(layers=tuple(zip(self.layout.layer_ids, self.layout.num_embeddings)),
                                     head_dim=32, model_revision=self.package["model_revision"], hash_abi=self.abi),
                    self.package["dataset_id"], self.package["generation"])
        views, samples = {}, {}
        for spec in validated.manifest.tensors:
            view = torch.arange(spec.shape[0] * spec.shape[1], dtype=torch.int64).to(torch.uint8).reshape(spec.shape)
            rows = sorted({0, spec.shape[0] // 2, spec.shape[0] - 1})
            views[spec.name] = view
            samples[spec.name] = dict(row_ids=rows, sha256=digest(view[rows].numpy().tobytes()))
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            self.assertTrue(b._verify_samples(views, validated, samples, "cpu"))
            synchronize.assert_called_once_with("cpu")
        first = validated.manifest.tensors[0].name
        views[first][0, 0] ^= 1
        with mock.patch.object(torch.cuda, "synchronize"):
            with self.assertRaisesRegex(b.BootstrapError, "GPU_READ_SAMPLE_MISMATCH"):
                b._verify_samples(views, validated, samples, "cpu")


if __name__ == "__main__":
    unittest.main()
