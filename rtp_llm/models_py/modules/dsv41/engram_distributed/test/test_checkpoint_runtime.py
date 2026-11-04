# SPDX-License-Identifier: Apache-2.0
"""CPU checkpoint and runtime attach integration with native doubles."""

import hashlib
import importlib
import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
import traceback
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41.engram_distributed import checkpoint
from rtp_llm.models_py.modules.dsv41.engram_distributed import manifest as m
from rtp_llm.models_py.modules.dsv41.engram_distributed import runtime as b
from rtp_llm.models_py.modules.dsv41.engram_distributed.checkpoint import load_rows
from rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_layout import (
    descriptors,
    prepare,
)


class DependencyFreeTest(unittest.TestCase):
    def test_duplicate_json_and_nonfinite_constants_are_rejected(self):
        for data in (b'{"weight_map":{},"weight_map":{}}', b'{"x":NaN}'):
            with self.assertRaises(checkpoint.CheckpointError):
                checkpoint._json(data)

    def test_unsafe_paths_are_rejected(self):
        for relative in (
            "/tmp/x.safetensors",
            "../x.safetensors",
            "a/../x.safetensors",
            "a//x.safetensors",
            "./x.safetensors",
            "x\\y.safetensors",
            "x\x00.safetensors",
            "model.bin",
            "",
        ):
            with self.subTest(relative=relative), self.assertRaises(
                checkpoint.CheckpointError
            ):
                checkpoint._parts(relative)


class RawReaderTest(unittest.TestCase):
    def test_short_reads_are_completed_and_eof_is_rejected(self):
        class Fragmented(io.BytesIO):
            def read(self, size):
                return super().read(min(size, 3))

        source = Fragmented(bytes(range(64)))
        self.assertEqual(checkpoint._read_exact(source, 63), bytes(range(63)))
        with self.assertRaises(checkpoint.CheckpointError) as error:
            checkpoint._read_exact(source, 2)
        self.assertEqual(error.exception.code, "SHORT_READ")

    def test_validated_header_drives_nonzero_offset_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "sample.safetensors"
            header = b'{"test":"header"}' + b" " * 23
            payload = bytes(range(97))
            path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
            _, start, snapshot = checkpoint._header(path, path.name)
            with checkpoint._open_tensor_source(root, snapshot) as (
                source,
                actual_start,
            ):
                self.assertEqual(actual_start, start)
                source.seek(actual_start + 31)
                self.assertEqual(checkpoint._read_exact(source, 66), payload[31:])
            with self.assertRaises(checkpoint.CheckpointError) as error:
                with checkpoint._open_tensor_source(
                    root, replace(snapshot, header_sha256="0" * 64)
                ):
                    self.fail("changed header must not yield")
            self.assertEqual(error.exception.code, "CHECKPOINT_CHANGED")

    def test_open_descriptor_detects_mutation_after_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "sample.safetensors"
            header = b'{"test":"header"}'
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"payload")
            _, _, snapshot = checkpoint._header(path, path.name)
            with self.assertRaises(checkpoint.CheckpointError) as error:
                with checkpoint._open_tensor_source(root, snapshot) as (source, _):
                    checkpoint._read_exact(source, 7)
                    with path.open("ab") as writer:
                        writer.write(b"changed")
            self.assertEqual(error.exception.code, "CHECKPOINT_CHANGED")

    def test_sparse_file_seek_above_four_gib_is_not_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offset.bin"
            offset = (4 << 30) + 123
            with path.open("wb") as source:
                source.seek(offset)
                source.write(b"large-offset-tail")
            with path.open("rb", buffering=0) as source:
                source.seek(offset)
                self.assertEqual(
                    checkpoint._read_exact(source, 17), b"large-offset-tail"
                )

    def test_cache_configuration_is_explicit_and_dynamic(self):
        with patch.object(
            checkpoint.os,
            "sysconf",
            side_effect=AssertionError("default must not query"),
        ):
            self.assertIsNone(checkpoint._cache_page_size(False))
        with patch.object(
            checkpoint.os, "posix_fadvise", lambda *args: None, create=True
        ), patch.object(
            checkpoint.os, "POSIX_FADV_DONTNEED", 4, create=True
        ), patch.object(
            checkpoint.os, "sysconf", return_value=65536
        ) as query:
            self.assertEqual(checkpoint._cache_page_size(True), 65536)
            query.assert_called_once_with("SC_PAGESIZE")
        for invalid in (1, "yes", None):
            with self.assertRaises(checkpoint.CheckpointError) as error:
                checkpoint._cache_page_size(invalid)
            self.assertEqual(error.exception.code, "INVALID_ARGUMENT")
        with patch.object(checkpoint.os, "posix_fadvise", None, create=True):
            with self.assertRaises(checkpoint.CheckpointError) as error:
                checkpoint._cache_page_size(True)
            self.assertEqual(error.exception.code, "CACHE_ADVICE_UNSUPPORTED")

    def test_discard_range_contains_only_fully_consumed_pages(self):
        with tempfile.TemporaryFile() as source, patch.object(
            checkpoint.os, "posix_fadvise", create=True
        ) as advise, patch.object(checkpoint.os, "POSIX_FADV_DONTNEED", 4, create=True):
            self.assertEqual(
                checkpoint._discard_consumed_pages(source, 1, 131073, 65536), 131072
            )
            advise.assert_called_once_with(source.fileno(), 65536, 65536, 4)
            advise.reset_mock()
            self.assertEqual(
                checkpoint._discard_consumed_pages(source, 131072, 131073, 65536),
                131072,
            )
            advise.assert_not_called()

    def test_discard_error_has_explicit_code(self):
        with tempfile.TemporaryFile() as source, patch.object(
            checkpoint.os,
            "posix_fadvise",
            side_effect=OSError("advice failed"),
            create=True,
        ), patch.object(checkpoint.os, "POSIX_FADV_DONTNEED", 4, create=True):
            with self.assertRaises(checkpoint.CheckpointError) as error:
                checkpoint._discard_consumed_pages(source, 0, 4096, 4096)
            self.assertEqual(error.exception.code, "CACHE_ADVICE_FAILED")


class CheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Missing real runtime dependencies are an error for this suite, not a skip.
        global torch, save_file
        import torch
        from safetensors.torch import save_file

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.layout = checkpoint.ExpectedLayout(
            layers=((1, 5), (2, 131)),
            head_dim=64,
            model_revision="loader-test-v1",
            hash_abi="sha256:" + "a" * 64,
        )
        self.index, self.payloads = {}, {}
        self.write_checkpoint()

    def write_index(self):
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": self.index})
        )

    def write_checkpoint(self, *, scale_fp8=False, weight_dtype=None):
        shards = ({}, {})
        self.index, self.payloads = {}, {}
        for layer, rows in self.layout.layers:
            for role, columns in (("weight", 64), ("scale", 2)):
                name = f"layers.{layer}.engram.embed.{role}"
                raw = (
                    (torch.arange(rows * columns, dtype=torch.int64) % 256)
                    .to(torch.uint8)
                    .reshape(rows, columns)
                )
                dtype = (
                    (weight_dtype or torch.float8_e4m3fn)
                    if role == "weight"
                    else (torch.float8_e8m0fnu if scale_fp8 else torch.uint8)
                )
                byte_dtypes = (
                    torch.uint8,
                    torch.float8_e4m3fn,
                    getattr(torch, "float8_e8m0fnu", None),
                )
                tensor = raw.view(dtype) if dtype in byte_dtypes else raw.to(dtype)
                shards[0 if role == "weight" else 1][name] = tensor
                self.payloads[name] = tensor.view(torch.uint8).numpy().tobytes()
                self.index[name] = (
                    "weights.safetensors" if role == "weight" else "scales.safetensors"
                )
        for filename, tensors in zip(
            ("weights.safetensors", "scales.safetensors"), shards
        ):
            save_file(tensors, self.root / filename)
        self.write_index()

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(checkpoint.CheckpointError) as error:
            function(*args, **kwargs)
        self.assertEqual(error.exception.code, code)

    def test_missing_layer_and_wrong_index_shard_are_rejected(self):
        self.index.pop("layers.2.engram.embed.scale")
        self.write_index()
        self.assert_code("MISSING_TENSOR", self.inspect)
        self.write_checkpoint()
        self.index["layers.1.engram.embed.weight"] = "scales.safetensors"
        self.write_index()
        with self.assertRaises(Exception):
            self.inspect()

    def test_checkpoint_paths_cannot_escape_or_traverse_symlinks(self):
        self.index["layers.1.engram.embed.weight"] = "../weights.safetensors"
        self.write_index()
        self.assert_code("UNSAFE_INDEX_PATH", self.inspect)
        self.write_checkpoint()
        (self.root / "alias.safetensors").symlink_to(self.root / "weights.safetensors")
        self.index["layers.1.engram.embed.weight"] = "alias.safetensors"
        self.write_index()
        self.assert_code("UNSAFE_INDEX_PATH", self.inspect)

    def test_wrong_dtype_and_shape_are_rejected(self):
        self.write_checkpoint(weight_dtype=torch.float16)
        self.assert_code("UNSUPPORTED_DTYPE", self.inspect)
        self.write_checkpoint()
        wrong = checkpoint.ExpectedLayout(
            layers=((1, 6), (2, 131)),
            head_dim=64,
            model_revision=self.layout.model_revision,
            hash_abi=self.layout.hash_abi,
        )
        self.assert_code(
            "SHAPE_MISMATCH", checkpoint.inspect_checkpoint, self.root, wrong
        )

    def test_lookup_width_and_manifest_tensor_count_fail_in_preflight(self):
        for layout, code in (
            (
                checkpoint.ExpectedLayout(
                    layers=self.layout.layers,
                    head_dim=96,
                    model_revision=self.layout.model_revision,
                    hash_abi=self.layout.hash_abi,
                ),
                "UNSUPPORTED_LOOKUP_WIDTH",
            ),
            (
                checkpoint.ExpectedLayout(
                    layers=tuple((i, 1) for i in range(513)),
                    head_dim=64,
                    model_revision=self.layout.model_revision,
                    hash_abi=self.layout.hash_abi,
                ),
                "MANIFEST_LIMIT",
            ),
        ):
            self.assert_code(code, checkpoint.inspect_checkpoint, self.root, layout)

    def inspect(self):
        return checkpoint.inspect_checkpoint(self.root, self.layout)

    def test_exact_shard_rows_keep_fp8_scale_bits_and_bounded_chunks(self):
        for fp8 in (False, True):
            self.write_checkpoint(scale_fp8=fp8)
            plan = self.inspect()
            snapshots = {s.relative_path: s for s in plan.files}
            for layer in plan.layers:
                for tensor in layer.tensors:
                    width = tensor.shape[1]
                    start, end = 1, tensor.shape[0] - 1
                    calls = []

                    class Owner:
                        def load(self, data, offset):
                            calls.append((offset, bytes(data)))

                    load_rows(
                        Owner(),
                        self.root,
                        tensor,
                        snapshots[tensor.relative_path],
                        start,
                        end,
                        width,
                        chunk_bytes=2 * width,
                    )
                    expected = self.payloads[tensor.name][start * width : end * width]
                    self.assertEqual(b"".join(data for _, data in calls), expected)
                    self.assertTrue(all(len(data) <= 2 * width for _, data in calls))
                    self.assertEqual(
                        [offset for offset, _ in calls],
                        list(range(0, len(expected), 2 * width)),
                    )

    def test_source_replacement_after_inspection_fails_before_native_write(self):
        plan = self.inspect()
        tensor = plan.layers[0].tensors[0]
        snapshot = next(
            s for s in plan.files if s.relative_path == tensor.relative_path
        )
        path = self.root / tensor.relative_path
        replacement = self.root / "replacement"
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        owner = unittest.mock.Mock()
        with self.assertRaisesRegex(checkpoint.CheckpointError, "CHECKPOINT_CHANGED"):
            load_rows(owner, self.root, tensor, snapshot, 0, 1, tensor.shape[1])
        owner.load.assert_not_called()

    def test_duplicate_header_or_truncated_payload_is_rejected(self):
        for duplicate in (True, False):
            self.write_checkpoint()
            path = self.root / "weights.safetensors"
            raw = path.read_bytes()
            if duplicate:
                length = int.from_bytes(raw[:8], "little")
                header = (
                    b'{"__metadata__":{},"__metadata__":{},'
                    + raw[8 : 8 + length].lstrip()[1:]
                )
                header += b" " * (-len(header) % 8)
                path.write_bytes(
                    len(header).to_bytes(8, "little") + header + raw[8 + length :]
                )
            else:
                path.write_bytes(raw[:-1])
            with self.assertRaises(Exception):
                self.inspect()

    def test_self_publish_loads_only_each_shard_and_rejects_changed_index(self):
        import sys
        from types import SimpleNamespace as NS
        from uuid import uuid4

        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            checkpoint as publisher,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed import (
            runtime as bootstrap,
        )
        from rtp_llm.models_py.modules.dsv41.engram_distributed.manifest import (
            prepare_manifest,
        )

        self.layout = checkpoint.ExpectedLayout(
            layers=((1, 65), (2, 131)),
            head_dim=64,
            model_revision="real-publish-test",
            hash_abi="sha256:" + "a" * 64,
        )
        self.write_checkpoint()
        (self.root / "config.json").write_text("{}")
        layout = NS(layer_ids=(1, 2), num_embeddings=(65, 131), head_dim=64)
        model = NS(
            EngramLayout=lambda _: layout,
            NgramHashState=lambda *a: None,
            reclaim_stale_shared_tables=lambda: None,
        )
        created = []
        mutate = [False]

        class Owner:
            def __init__(self, size, *args):
                self.data = bytearray(size)
                self.closed = False
                self.handle = uuid4().bytes
                created.append(self)

            def load(self, raw, offset):
                self.data[offset : offset + len(raw)] = raw
                if mutate[0]:
                    mutate[0] = False
                    path = self_root / "model.safetensors.index.json"
                    path.write_text(path.read_text() + " ")

            def export_handle(self):
                return self.handle

            def close(self):
                self.closed = True

        self_root = self.root
        native = NS(
            FabricAllocation=Owner,
            host_numa_granularity=lambda *a: 64,
            max_allocation_bytes=1 << 30,
            max_load_chunk_bytes=128,
        )
        generation = str(uuid4())
        with patch.dict(
            sys.modules, {"rtp_llm.models_py.modules.dsv41.engram": model}
        ), patch.object(publisher, "_load_native", lambda: native), patch.object(
            publisher, "hash_state_digest", lambda _: self.layout.hash_abi
        ):
            documents = [
                publisher.shard_publisher.publish(
                    self.root,
                    self.layout.model_revision,
                    f"host-{part}",
                    generation,
                    part,
                    2,
                )
                for part in range(2)
            ]
            parsed, parsed_handles = prepare_manifest(
                documents,
                2,
                generation,
                self.layout.model_revision,
                self.layout.hash_abi,
            )
            samples = publisher.checkpoint_samples(self.root, parsed.manifest)
            for tensor in parsed.manifest.tensors:
                name = tensor.name
                row_count, width = tensor.shape
                split = tensor.shards[0].row_end
                rows = sorted({0, row_count // 2, split - 1, split, row_count - 1})
                raw = self.payloads[name]
                sampled_bytes = b"".join(
                    raw[row * width : (row + 1) * width] for row in rows
                )
                self.assertEqual(
                    samples[name], dict(row_ids=rows, sha256=digest(sampled_bytes))
                )
                for part in tensor.shards:
                    expected = raw[part.row_begin * width : part.row_end * width]
                    owner = next(
                        o
                        for o in created
                        if o.handle == parsed_handles[part.allocation_id]
                    )
                    self.assertEqual(bytes(owner.data[: len(expected)]), expected)
            self.assertEqual(publisher.shard_publisher.release(generation), [])
            self.assertTrue(all(owner.closed for owner in created))
            created.clear()
            mutate[0] = True
            with self.assertRaisesRegex(
                checkpoint.CheckpointError, "CHECKPOINT_CHANGED"
            ):
                publisher.shard_publisher.publish(
                    self.root, self.layout.model_revision, "host-0", generation, 0, 2
                )
            self.assertTrue(created and all(owner.closed for owner in created))
            self.assertNotIn(generation, publisher.shard_publisher._owners)


try:
    import torch
except ImportError:
    torch = None


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def real_hash_state():
    e = importlib.import_module("rtp_llm.models_py.modules.dsv41.engram")
    config = dict(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[128, 256],
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=8,
        engram_pad_token_id=2,
        engram_vocab_size=5,
    )
    return e.NgramHashState(e.EngramLayout(config), token_map=list(range(8)))


class NativeLoaderTest(unittest.TestCase):
    def setUp(self):
        self.ops = types.ModuleType("rtp_llm.ops")
        self.ops.load_engram_ops = mock.Mock()
        patcher = mock.patch.dict(sys.modules, {"rtp_llm.ops": self.ops})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_returns_the_prebuilt_module(self):
        native = object()
        with mock.patch.object(
            self.ops, "load_engram_ops", return_value=native
        ) as load:
            self.assertIs(b._load_native(), native)
        load.assert_called_once_with()

    def test_broken_logging_handler_preserves_loader_error(self):
        class BrokenHandler(logging.Handler):
            def emit(self, record):
                raise RuntimeError("private logging backend error")

        logger = logging.getLogger(b._load_native.__module__)
        handler = BrokenHandler(level=logging.ERROR)
        logger.addHandler(handler)
        try:
            with mock.patch.object(
                self.ops, "load_engram_ops", side_effect=ImportError("private symbol")
            ):
                with self.assertRaisesRegex(
                    b.BootstrapError, "^NATIVE_MODULE_LOAD_FAILED$"
                ):
                    b._load_native()
        finally:
            logger.removeHandler(handler)

    def test_native_load_categories_preserve_the_local_loader_cause(self):
        secret = "/private/checkpoint/handle-value"
        cases = (
            (
                ModuleNotFoundError(secret, name="rtp_llm.ops"),
                "NATIVE_MODULE_DEPENDENCY_MISSING",
                "dependency_missing",
            ),
            (
                ModuleNotFoundError(secret, name="libfabric_memory"),
                "PREBUILT_ENGRAM_NATIVE_MODULE_REQUIRED",
                "module_missing",
            ),
            (
                ModuleNotFoundError(secret, name="private_native_dependency"),
                "NATIVE_MODULE_DEPENDENCY_MISSING",
                "dependency_missing",
            ),
            (
                ModuleNotFoundError(secret),
                "NATIVE_MODULE_LOAD_FAILED",
                "import_failure",
            ),
            (
                ImportError(secret + ": libcuda.so.1: cannot open shared object file"),
                "NATIVE_MODULE_LOAD_FAILED",
                "import_failure",
            ),
            (
                ImportError(secret + ": undefined symbol: private_symbol"),
                "NATIVE_MODULE_LOAD_FAILED",
                "import_failure",
            ),
            (
                OSError(secret + ": cannot load shared library"),
                "NATIVE_MODULE_LOAD_FAILED",
                "shared_library_failure",
            ),
        )
        for original, code, category in cases:
            with self.subTest(code=code, category=category):
                with mock.patch.object(
                    self.ops, "load_engram_ops", side_effect=original
                ):
                    with self.assertLogs(
                        b._load_native.__module__, level="ERROR"
                    ) as captured:
                        try:
                            b._load_native()
                        except b.BootstrapError as error:
                            self.assertEqual(error.code, code)
                            self.assertIs(error.__cause__, original)
                            rendered = "".join(
                                traceback.format_exception(
                                    type(error), error, error.__traceback__
                                )
                            )
                        else:
                            self.fail("native import unexpectedly succeeded")
                self.assertEqual(len(captured.records), 1)
                self.assertIn("event=native_load_failed", captured.output[0])
                self.assertIn("error_class=" + category, captured.output[0])
                self.assertIn("error_code=" + code, captured.output[0])
                self.assertIsNone(captured.records[0].exc_info)
                self.assertIn(str(original), rendered)
                self.assertNotIn(secret, captured.output[0])


class HashSemanticTest(unittest.TestCase):
    def test_real_hash_state_is_deterministic(self):
        self.assertEqual(
            b.hash_state_digest(real_hash_state()),
            b.hash_state_digest(real_hash_state()),
        )

    def test_every_hash_constant_changes_digest(self):
        baseline = b.hash_state_digest(real_hash_state())
        for name in ("token_map", "multipliers", "primes", "offsets"):
            with self.subTest(constant=name):
                state = real_hash_state()
                tensor = (
                    state.layout.offsets if name == "offsets" else getattr(state, name)
                )
                tensor.reshape(-1)[-1] += 1
                self.assertNotEqual(b.hash_state_digest(state), baseline)

    def test_pad_and_layout_semantics_change_digest(self):
        baseline = b.hash_state_digest(real_hash_state())
        for name, value in (
            ("pad_id", 3),
            ("pad_token_id", 3),
            ("max_ngram_size", 4),
            ("head_dim", 64),
            ("n_heads", 3),
            ("compressed_vocab_size", 9),
            ("layer_ids", (1, 15)),
            ("num_embeddings", (129, 256)),
        ):
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
        for device, dtype in (
            ("meta", torch.int64),
            ("cpu", torch.int32),
            ("cpu", torch.float32),
        ):
            state = real_hash_state()
            state.token_map = torch.empty(
                state.token_map.shape, dtype=dtype, device=device
            )
            with self.assertRaisesRegex(
                b.BootstrapError, "HASH_CONSTANTS_MUST_BE_CPU_INT64"
            ):
                b.hash_state_digest(state)


@unittest.skipIf(torch is None, "CPU PyTorch required; native/GPU calls are mocked")
class AttachFailureTest(unittest.TestCase):
    def setUp(self):
        self.hash_state = real_hash_state()
        self.layout = self.hash_state.layout
        self.abi = b.hash_state_digest(self.hash_state)
        docs = descriptors(
            2,
            self.layout.head_dim,
            layers=tuple(zip(self.layout.layer_ids, self.layout.num_embeddings)),
            granularity=1,
            hash_abi=self.abi,
        )
        self.parsed, handles = prepare(docs, hash_abi=self.abi)
        manifest = self.parsed.manifest
        self.package = dict(
            manifest=self.parsed,
            generation=manifest.generation,
            model_revision=manifest.model_revision,
            capabilities=handles,
            samples={
                tensor.name: {
                    "row_ids": sorted(
                        {0, tensor.shape[0] // 2, tensor.shape[0] - 1}
                        | {
                            row
                            for part in tensor.shards
                            for row in (part.row_begin, part.row_end - 1)
                        }
                    ),
                    "sha256": "sha256:" + "c" * 64,
                }
                for tensor in manifest.tensors
            },
        )
        self.storage = mock.Mock()
        self.storage.verify.return_value = None
        self.storage.close.return_value = None
        self.native = types.SimpleNamespace(max_allocation_bytes=128 << 30)
        self.retained = []
        self.native_patch = mock.patch.object(
            b, "_load_native", return_value=self.native
        )
        self.storage_patch = mock.patch.object(
            b.FabricEngramStorage, "_attach_validated", return_value=self.storage
        )
        self.runtime = b.EngramRuntime()
        self.retain_patch = mock.patch.object(
            self.runtime, "retain", side_effect=self.retained.append
        )
        self.native_mock = self.native_patch.start()
        self.attach_mock = self.storage_patch.start()
        self.retain_patch.start()
        self.addCleanup(self.native_patch.stop)
        self.addCleanup(self.storage_patch.stop)
        self.addCleanup(self.retain_patch.stop)

    def do_attach(self, device="cuda:2", environ=None):
        from rtp_llm.models_py.modules.dsv41.engram_distributed.config import (
            EngramOptions,
            ShardConfig,
            StoragePolicy,
        )

        env = {} if environ is None else environ
        options = EngramOptions(
            StoragePolicy.from_environ(env),
            ShardConfig.from_environ(env, inferred_count=2),
            "eager",
        )
        return self.runtime.attach_from_package(
            self.package,
            layout=self.layout,
            hash_state=self.hash_state,
            device=device,
            options=options,
        )

    def test_explicit_p_mismatch_rejected_before_native_load(self):
        with self.assertRaisesRegex(ValueError, "COUNT_MISMATCH"):
            self.do_attach(
                environ={"RTP_ENGRAM_STORAGE": "host_shared", "ENGRAM_SHARD_COUNT": "3"}
            )
        self.native_mock.assert_not_called()
        self.attach_mock.assert_not_called()

    def test_native_capacity_mismatch_rejects_before_first_import(self):
        for maximum in (None, True, 0, 1):
            self.native.max_allocation_bytes = maximum
            with self.assertRaisesRegex(b.BootstrapError, "NATIVE_ALLOCATION_LIMIT"):
                self.do_attach()
            self.attach_mock.assert_not_called()

    def test_local_policy_does_not_limit_remote_mapping_and_metadata_is_reused(
        self,
    ):
        env = {
            "RTP_ENGRAM_MAX_ALLOCATION_BYTES": "8192",
            "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": "8448",
        }
        self.do_attach(environ=env)
        validated = self.attach_mock.call_args.args[0]
        self.assertIsInstance(validated, m.ValidatedManifest)
        self.assertIs(validated.manifest, self.parsed.manifest)
        self.assertGreater(
            validated.physical_bytes, int(env["RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES"])
        )
        self.assertEqual(
            validated.layout.layers,
            tuple(zip(self.layout.layer_ids, self.layout.num_embeddings)),
        )

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
        with self.assertRaisesRegex(m.ManifestError, "MODEL_CONTRACT_MISMATCH"):
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
        validated = m.bind_manifest(
            self.parsed,
            m.ExpectedLayout(
                layers=tuple(zip(self.layout.layer_ids, self.layout.num_embeddings)),
                head_dim=32,
                model_revision=self.package["model_revision"],
                hash_abi=self.abi,
            ),
            self.package["generation"],
            m.Limits(),
        )
        views, samples = {}, {}
        for spec in validated.manifest.tensors:
            view = (
                torch.arange(spec.shape[0] * spec.shape[1], dtype=torch.int64)
                .to(torch.uint8)
                .reshape(spec.shape)
            )
            rows = sorted({0, spec.shape[0] // 2, spec.shape[0] - 1})
            views[spec.name] = view
            samples[spec.name] = dict(
                row_ids=rows, sha256=digest(view[rows].numpy().tobytes())
            )
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
