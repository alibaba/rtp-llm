# SPDX-License-Identifier: Apache-2.0
"""CPU tests with real torch/safetensors and resource-accounting FakeNative.

The real runtime dependencies are mandatory: missing wheels fail this suite
instead of reporting success after skipping the checkpoint loader tests.
"""
import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from rtp_llm.models_py.modules.dsv41 import engram_checkpoint as m


class FakeNative:
    max_allocation_bytes = 1 << 30
    max_load_chunk_bytes = 64 << 20

    def __init__(self, granularity=4096):
        self.granularity = granularity
        self.owners, self.events, self.close_attempts = [], [], []
        self.fail_create = self.fail_load = self.fail_export = None
        self.load_error = RuntimeError
        self.fail_close = set()
        self.load_hook = None
        self.load_calls = self.query_calls = 0

    def host_numa_granularity(self, numa, device):
        self.query_calls += 1
        return self.granularity

    def Owner(self, size, numa, device):
        index = len(self.owners)
        if index == self.fail_create:
            raise RuntimeError("injected create failure")
        result = FakeOwner(self, index, size)
        self.owners.append(result)
        self.events.append(("create", index))
        return result


class FakeOwner:
    def __init__(self, native, index, size):
        self.native, self.index, self.size = native, index, size
        self.padded = (size + native.granularity - 1) // native.granularity * native.granularity
        self.data = bytearray(self.padded)
        self.closed = self.sealed = False

    def info(self):
        return dict(allocation_bytes=self.padded, granularity=self.native.granularity)

    def load(self, data, offset=0):
        assert not self.closed and not self.sealed
        assert len(data) <= self.native.max_load_chunk_bytes
        assert 0 <= offset <= offset + len(data) <= self.size
        self.native.load_calls += 1
        if self.native.load_hook:
            self.native.load_hook(self.native.load_calls)
        if self.native.load_calls == self.native.fail_load:
            raise self.native.load_error("injected load failure")
        self.data[offset:offset + len(data)] = data
        self.native.events.append(("load", self.index, offset, len(data)))

    def export_handle(self):
        assert not self.closed
        if self.index == self.native.fail_export:
            raise RuntimeError("injected export failure")
        self.sealed = True
        self.native.events.append(("export", self.index))
        return f"private-capability-{self.index}".encode()

    def close(self):
        self.native.close_attempts.append(self.index)
        if self.index in self.native.fail_close:
            raise RuntimeError("injected close failure")
        self.closed = True


class DependencyFreeTest(unittest.TestCase):
    def test_duplicate_json_and_nonfinite_constants_are_rejected(self):
        for data in (b'{"weight_map":{},"weight_map":{}}', b'{"x":NaN}'):
            with self.assertRaises(m.CheckpointError):
                m._json(data)

    def test_unsafe_paths_are_rejected(self):
        for relative in ("/tmp/x.safetensors", "../x.safetensors", "a/../x.safetensors",
                         "a//x.safetensors", "./x.safetensors", "x\\y.safetensors",
                         "x\x00.safetensors", "model.bin", ""):
            with self.subTest(relative=relative), self.assertRaises(m.CheckpointError):
                m._parts(relative)

    def test_pending_generation_never_exposes_capabilities_or_repr(self):
        generation = m.LoadedGeneration()
        generation._handles["id"] = b"secret"
        self.assertNotIn("secret", repr(generation))
        with self.assertRaises(m.CheckpointError):
            generation.private_handles
        generation.close()
        generation.close()


class RawReaderTest(unittest.TestCase):
    def test_short_reads_are_completed_and_eof_is_rejected(self):
        class Fragmented(io.BytesIO):
            def read(self, size):
                return super().read(min(size, 3))
        source = Fragmented(bytes(range(64)))
        self.assertEqual(m._read_exact(source, 63), bytes(range(63)))
        with self.assertRaises(m.CheckpointError) as error:
            m._read_exact(source, 2)
        self.assertEqual(error.exception.code, "SHORT_READ")

    def test_validated_header_drives_nonzero_offset_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "sample.safetensors"
            header = b'{"test":"header"}' + b" " * 23
            payload = bytes(range(97))
            path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
            _, start, snapshot = m._header(path, path.name)
            with m._open_tensor_source(root, snapshot) as (source, actual_start):
                self.assertEqual(actual_start, start)
                source.seek(actual_start + 31)
                self.assertEqual(m._read_exact(source, 66), payload[31:])
            with self.assertRaises(m.CheckpointError) as error:
                with m._open_tensor_source(root, replace(snapshot, header_sha256="0" * 64)):
                    self.fail("changed header must not yield")
            self.assertEqual(error.exception.code, "CHECKPOINT_CHANGED")

    def test_open_descriptor_detects_mutation_after_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "sample.safetensors"
            header = b'{"test":"header"}'
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"payload")
            _, _, snapshot = m._header(path, path.name)
            with self.assertRaises(m.CheckpointError) as error:
                with m._open_tensor_source(root, snapshot) as (source, _):
                    m._read_exact(source, 7)
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
                self.assertEqual(m._read_exact(source, 17), b"large-offset-tail")

    def test_cache_configuration_is_explicit_and_dynamic(self):
        with patch.object(m.os, "sysconf", side_effect=AssertionError("default must not query")):
            self.assertIsNone(m._cache_page_size(False))
        with patch.object(m.os, "posix_fadvise", lambda *args: None, create=True), \
                patch.object(m.os, "POSIX_FADV_DONTNEED", 4, create=True), \
                patch.object(m.os, "sysconf", return_value=65536) as query:
            self.assertEqual(m._cache_page_size(True), 65536)
            query.assert_called_once_with("SC_PAGESIZE")
        for invalid in (1, "yes", None):
            with self.assertRaises(m.CheckpointError) as error:
                m._cache_page_size(invalid)
            self.assertEqual(error.exception.code, "INVALID_ARGUMENT")
        with patch.object(m.os, "posix_fadvise", None, create=True):
            with self.assertRaises(m.CheckpointError) as error:
                m._cache_page_size(True)
            self.assertEqual(error.exception.code, "CACHE_ADVICE_UNSUPPORTED")

    def test_discard_range_contains_only_fully_consumed_pages(self):
        with tempfile.TemporaryFile() as source, \
                patch.object(m.os, "posix_fadvise", create=True) as advise, \
                patch.object(m.os, "POSIX_FADV_DONTNEED", 4, create=True):
            self.assertEqual(m._discard_consumed_pages(source, 1, 131073, 65536), 131072)
            advise.assert_called_once_with(source.fileno(), 65536, 65536, 4)
            advise.reset_mock()
            self.assertEqual(m._discard_consumed_pages(source, 131072, 131073, 65536), 131072)
            advise.assert_not_called()

    def test_discard_error_has_explicit_code(self):
        with tempfile.TemporaryFile() as source, \
                patch.object(m.os, "posix_fadvise", side_effect=OSError("advice failed"), create=True), \
                patch.object(m.os, "POSIX_FADV_DONTNEED", 4, create=True):
            with self.assertRaises(m.CheckpointError) as error:
                m._discard_consumed_pages(source, 0, 4096, 4096)
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
        self.root = Path(self.temp.name)
        self.layout = m.ExpectedLayout(layers=((1, 5), (2, 131)), head_dim=64,
                                       model_revision="loader-test-v1", hash_abi="sha256:" + "a" * 64)
        self.index, self.payloads = {}, {}
        self.write_checkpoint()

    def write_index(self):
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": self.index}))

    def write_checkpoint(self, *, scale_fp8=False, weight_dtype=None):
        shards = ({}, {})
        self.index, self.payloads = {}, {}
        for layer, rows in self.layout.layers:
            for role, columns in (("weight", 64), ("scale", 2)):
                name = f"layers.{layer}.engram.embed.{role}"
                raw = (torch.arange(rows * columns, dtype=torch.int64) % 256).to(torch.uint8).reshape(rows, columns)
                dtype = (weight_dtype or torch.float8_e4m3fn) if role == "weight" else (
                    torch.float8_e8m0fnu if scale_fp8 else torch.uint8)
                byte_dtypes = (torch.uint8, torch.float8_e4m3fn, getattr(torch, "float8_e8m0fnu", None))
                tensor = raw.view(dtype) if dtype in byte_dtypes else raw.to(dtype)
                shards[0 if role == "weight" else 1][name] = tensor
                self.payloads[name] = tensor.view(torch.uint8).numpy().tobytes()
                self.index[name] = "weights.safetensors" if role == "weight" else "scales.safetensors"
        for filename, tensors in zip(("weights.safetensors", "scales.safetensors"), shards):
            save_file(tensors, self.root / filename)
        self.write_index()

    def inspect(self, **changes):
        options = dict(allocation_granularity=4096, budget_bytes=16384)
        options.update(changes)
        return m.inspect_checkpoint(self.root, self.layout, **options)

    def load(self, native=None, plan=None, **changes):
        options = dict(native=native or FakeNative(), device=0, numa=0, chunk_bytes=128)
        options.update(changes)
        return m.load_checkpoint(plan or self.inspect(), **options)

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(m.CheckpointError) as error:
            function(*args, **kwargs)
        self.assertEqual(error.exception.code, code)

    def test_multifile_complete_generation_full_hashes_and_bounded_chunks(self):
        native = FakeNative()
        plan = self.inspect()
        self.assertEqual(plan.physical_bytes, 16384)
        result = self.load(native, plan)
        self.assertEqual(len(native.owners), 2)
        self.assertEqual(len(result.private_handles), 2)
        self.assertNotIn("private-capability", repr(result))
        with self.assertRaises(TypeError):
            result.private_handles["new"] = b"cannot mutate"
        for tensor in result.manifest.tensors:
            part = tensor.shards[0]
            layer = int(tensor.name.split(".")[1])
            owner = native.owners[layer - 1]
            payload = bytes(owner.data[part.byte_offset:part.byte_offset + part.logical_bytes])
            self.assertEqual(payload, self.payloads[tensor.name])
            self.assertEqual(part.sha256, "sha256:" + hashlib.sha256(payload).hexdigest())
        self.assertTrue(all(event[3] <= 128 for event in native.events if event[0] == "load"))
        kinds = [event[0] for event in native.events]
        self.assertLess(max(i for i, kind in enumerate(kinds) if kind == "load"), kinds.index("export"))
        result.close()
        self.assertEqual(native.close_attempts, [1, 0])
        result.close()
        self.assertTrue(all(owner.closed for owner in native.owners))
        self.assert_code("GENERATION_NOT_READY", lambda: result.private_handles)

    def test_e8m0_scale_bits_are_preserved(self):
        self.write_checkpoint(scale_fp8=True)
        result = self.load()
        for tensor in result.manifest.tensors:
            self.assertEqual(tensor.shards[0].sha256,
                             "sha256:" + hashlib.sha256(self.payloads[tensor.name]).hexdigest())
        result.close()

    def test_stream_uses_raw_bytes_not_safetensors_tensor_materialization(self):
        import safetensors
        original = safetensors.safe_open
        calls = []
        def metadata_only(*args, **kwargs):
            calls.append(args[0])
            # Inspection needs two metadata readers; streaming must not open a third.
            if len(calls) > 2:
                raise AssertionError("stream must not mmap tensors")
            return original(*args, **kwargs)
        plan = self.inspect()
        with patch.object(safetensors, "safe_open", side_effect=metadata_only):
            result = self.load(plan=plan, chunk_bytes=129)
        self.assertEqual(len(calls), 2)
        for tensor in result.manifest.tensors:
            self.assertEqual(tensor.shards[0].sha256,
                             "sha256:" + hashlib.sha256(self.payloads[tensor.name]).hexdigest())
        result.close()

    def test_default_never_advises_file_cache(self):
        with patch.object(m.os, "posix_fadvise", side_effect=AssertionError("unexpected cache advice"), create=True):
            result = self.load()
            result.close()

    def test_raw_tensor_offset_above_four_gib_with_real_safetensors_metadata(self):
        prefix = 4 << 30
        header = {"unused": {"dtype": "U8", "shape": [prefix], "data_offsets": [0, prefix]}}
        offset = prefix
        for name, payload in self.payloads.items():
            layer = int(name.split(".")[1])
            rows = dict(self.layout.layers)[layer]
            columns = 64 if name.endswith(".weight") else 2
            header[name] = dict(dtype="F8_E4M3" if columns == 64 else "U8",
                                shape=[rows, columns], data_offsets=[offset, offset + len(payload)])
            offset += len(payload)
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path = self.root / "sparse.safetensors"
        with path.open("wb") as source:
            source.write(len(encoded).to_bytes(8, "little") + encoded)
            source.seek(8 + len(encoded) + prefix)
            for payload in self.payloads.values():
                source.write(payload)
        self.index = {name: path.name for name in self.payloads}
        self.write_index()
        result = self.load()
        for tensor in result.manifest.tensors:
            self.assertEqual(tensor.shards[0].sha256,
                             "sha256:" + hashlib.sha256(self.payloads[tensor.name]).hexdigest())
        result.close()

    def test_raw_short_read_rolls_back_before_any_export(self):
        original = m._read_exact
        def truncate_payload(source, size):
            if size == 128:
                raise m.CheckpointError("SHORT_READ")
            return original(source, size)
        native = FakeNative()
        with patch.object(m, "_read_exact", side_effect=truncate_payload):
            self.assert_code("SHORT_READ", self.load, native)
        self.assertEqual(native.close_attempts, [0])
        self.assertFalse(any(event[0] == "export" for event in native.events))

    def test_cache_preflight_failure_never_allocates(self):
        native = FakeNative()
        with patch.object(m, "_cache_page_size", return_value=32), \
                patch.object(m.os, "POSIX_FADV_DONTNEED", 4, create=True), \
                patch.object(m.os, "posix_fadvise", side_effect=OSError("cannot advise"), create=True):
            self.assert_code("CACHE_ADVICE_FAILED", self.load, native, discard_file_cache=True)
        self.assertEqual(native.owners, [])

    def test_payload_mutation_during_raw_stream_rolls_back(self):
        native = FakeNative()
        def mutate(call):
            if call == 1:
                with (self.root / "weights.safetensors").open("ab") as source:
                    source.write(b"changed")
        native.load_hook = mutate
        self.assert_code("CHECKPOINT_CHANGED", self.load, native)
        self.assertEqual(native.close_attempts, [0])
        self.assertFalse(any(event[0] == "export" for event in native.events))

    def test_private_cache_advice_precedes_allocation_and_follows_completed_load(self):
        native = FakeNative()
        plan = self.inspect()
        events = []
        original_discard = m._discard_consumed_pages
        def discard(source, start, end, page_size):
            events.append(("discard", Path(source.name).name, start, end, len(native.events)))
            return original_discard(source, start, end, page_size)
        calls = []
        def advise(fd, offset, length, advice):
            calls.append((offset, length, advice))
        with patch.object(m, "_cache_page_size", return_value=32), \
                patch.object(m, "_DISCARD_INTERVAL_BYTES", 256), \
                patch.object(m, "_discard_consumed_pages", side_effect=discard), \
                patch.object(m.os, "posix_fadvise", side_effect=advise, create=True), \
                patch.object(m.os, "POSIX_FADV_DONTNEED", 4, create=True):
            result = self.load(native, plan, discard_file_cache=True)
        self.assertEqual(len(events[:2]), len(plan.files))
        self.assertTrue(all(event[4] == 0 for event in events[:2]))
        self.assertEqual({event[1] for event in events[:2]}, {snapshot.relative_path for snapshot in plan.files})
        for event in events[2:]:
            previous = native.events[event[4] - 1]
            self.assertEqual(previous[0], "load")
            tensor = next(t for t in plan.layers[previous[1]].tensors if t.relative_path == event[1])
            _, data_start, _ = m._header(self.root / event[1], event[1])
            tensor_start = data_start + tensor.data_range[0]
            self.assertGreaterEqual(event[2], tensor_start)
            self.assertEqual(event[3], tensor_start + previous[2] - tensor.target_offset + previous[3])
            self.assertGreaterEqual(event[3], event[2])
        self.assertTrue(all(offset % 32 == length % 32 == 0 and length > 0
                            for offset, length, _ in calls))
        self.assertGreater(len(events), 4)
        result.close()

    def test_stream_cache_advice_failure_rolls_back_loaded_owner(self):
        native = FakeNative()
        original = m._discard_consumed_pages
        calls = 0
        def discard(*args):
            nonlocal calls
            calls += 1
            if calls == 3:  # Two private files before allocation, then consumed bytes.
                raise m.CheckpointError("CACHE_ADVICE_FAILED")
            return original(*args)
        with patch.object(m, "_cache_page_size", return_value=32), \
                patch.object(m, "_DISCARD_INTERVAL_BYTES", 256), \
                patch.object(m, "_discard_consumed_pages", side_effect=discard), \
                patch.object(m.os, "posix_fadvise", create=True), \
                patch.object(m.os, "POSIX_FADV_DONTNEED", 4, create=True):
            self.assert_code("CACHE_ADVICE_FAILED", self.load, native, discard_file_cache=True)
        self.assertEqual(native.close_attempts, [0])
        self.assertTrue(any(event[0] == "load" for event in native.events))
        self.assertFalse(any(event[0] == "export" for event in native.events))

    def test_budget_and_padded_allocation_limits_are_preallocation(self):
        self.assert_code("BUDGET_EXCEEDED", self.inspect, budget_bytes=16383)
        self.assert_code("ALLOCATION_LIMIT", self.inspect, max_allocation_bytes=12287)
        self.assertEqual(self.inspect(max_allocation_bytes=12288).physical_bytes, 16384)
        self.assert_code("INVALID_ARGUMENT", self.inspect, max_allocation_bytes=(128 << 30) + 1)

    def test_explicit_large_policy_preserves_small_fixture_and_default(self):
        default = self.inspect()
        self.assertEqual(default.max_allocation_bytes, 1 << 30)
        large = self.inspect(max_allocation_bytes=128 << 30, budget_bytes=256 << 30)
        self.assertEqual(large.layers, default.layers)
        self.assertEqual(large.physical_bytes, default.physical_bytes)
        native = FakeNative()
        loaded = self.load(native, large)
        self.assertEqual(loaded.physical_bytes, default.physical_bytes)
        loaded.close()

    def test_capacity_policy_rejected_before_source_access_or_owner(self):
        for options in (dict(budget_bytes=(256 << 30) + 1),
                        dict(max_allocation_bytes=True), dict(budget_bytes=False)):
            with patch.object(m, "_path", side_effect=AssertionError("must not access source")):
                self.assert_code("INVALID_ARGUMENT", self.inspect, **options)

    def test_native_granularity_and_limits_rechecked_before_owner(self):
        plan = self.inspect()
        native = FakeNative(granularity=8192)
        with self.assertRaises(m.CheckpointError):
            self.load(native, plan)
        self.assertEqual(native.owners, [])
        native = FakeNative()
        native.max_allocation_bytes = 4096
        self.assert_code("ALLOCATION_LIMIT", self.load, native, plan)
        self.assertEqual(native.owners, [])
        self.assert_code("CHUNK_SMALLER_THAN_ROW", self.load, native, plan, chunk_bytes=63)
        self.assert_code("INVALID_ARGUMENT", self.load, native, plan, chunk_bytes=(64 << 20) + 1)
        self.assertEqual(native.owners, [])

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
        wrong = m.ExpectedLayout(layers=((1, 6), (2, 131)), head_dim=64,
                                 model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi)
        self.assert_code("SHAPE_MISMATCH", m.inspect_checkpoint, self.root, wrong,
                         allocation_granularity=4096, budget_bytes=16384)

    def test_lookup_width_and_manifest_tensor_count_fail_in_preflight(self):
        for layout, code in (
                (m.ExpectedLayout(layers=self.layout.layers, head_dim=96,
                                  model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi),
                 "UNSUPPORTED_LOOKUP_WIDTH"),
                (m.ExpectedLayout(layers=tuple((i, 1) for i in range(513)), head_dim=64,
                                  model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi),
                 "MANIFEST_LIMIT")):
            self.assert_code(code, m.inspect_checkpoint, self.root, layout,
                             allocation_granularity=4096, budget_bytes=1 << 30)

    def test_inspect_then_same_size_replacement_is_rejected_before_owner(self):
        plan = self.inspect()
        path = self.root / "weights.safetensors"
        replacement = self.root / "replacement"
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        native = FakeNative()
        self.assert_code("CHECKPOINT_OR_GRANULARITY_CHANGED", self.load, native, plan)
        self.assertEqual(native.owners, [])

    def test_same_size_payload_write_with_restored_mtime_and_changed_ctime_is_rejected(self):
        plan = self.inspect()
        path = self.root / "weights.safetensors"
        original = path.stat()
        value = path.read_bytes()[-1:]
        # Linux filesystems can assign identical ctime to consecutive writes
        # within one clock tick, even when stat exposes a nanosecond field.
        # Exercise a real ctime-only identity change, not a timing assumption.
        deadline = time.monotonic() + 2
        while True:
            with path.open("r+b") as source:
                source.seek(-1, 2)
                source.write(bytes((value[0] ^ 1,)))
                source.flush()
                os.fsync(source.fileno())
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            changed = path.stat()
            if changed.st_ctime_ns != original.st_ctime_ns:
                break
            self.assertLess(time.monotonic(), deadline, "filesystem did not expose a ctime change")
            time.sleep(0.01)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
            self.assertEqual(getattr(changed, name), getattr(original, name))
        self.assertNotEqual(path.read_bytes()[-1:], value)
        native = FakeNative()
        self.assert_code("CHECKPOINT_OR_GRANULARITY_CHANGED", self.load, native, plan)
        self.assertEqual(native.owners, [])

    def test_duplicate_checkpoint_header_keys_rejected_before_owner(self):
        plan = self.inspect()
        path = self.root / "weights.safetensors"
        raw = path.read_bytes()
        length = int.from_bytes(raw[:8], "little")
        header = b'{"__metadata__":{},"__metadata__":{},' + raw[8:8 + length].lstrip()[1:]
        header += b" " * (-len(header) % 8)
        path.write_bytes(len(header).to_bytes(8, "little") + header + raw[8 + length:])
        native = FakeNative()
        self.assert_code("DUPLICATE_JSON_KEY", self.load, native, plan)
        self.assertEqual(native.owners, [])

    def test_corrupt_range_or_truncation_never_allocates_owner(self):
        for corrupt_range in (False, True):
            self.write_checkpoint()
            plan = self.inspect()
            path = self.root / "weights.safetensors"
            raw = path.read_bytes()
            if corrupt_range:
                length = int.from_bytes(raw[:8], "little")
                header = json.loads(raw[8:8 + length])
                header["layers.1.engram.embed.weight"]["data_offsets"][1] = len(raw) + 1
                encoded = json.dumps(header).encode()
                encoded += b" " * (-len(encoded) % 8)
                path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + raw[8 + length:])
            else:
                path.write_bytes(raw[:-1])
            native = FakeNative()
            with self.assertRaises(Exception):
                self.load(native, plan)
            self.assertEqual(native.owners, [])

    def test_index_or_payload_changes_during_load_abort_before_export(self):
        native = FakeNative()
        def change_index(call):
            if call == 1:
                path = self.root / "model.safetensors.index.json"
                path.write_text(path.read_text() + " ")
        native.load_hook = change_index
        self.assert_code("CHECKPOINT_CHANGED", self.load, native)
        self.assertFalse(any(event[0] == "export" for event in native.events))
        self.assertEqual(native.close_attempts, [1, 0])

    def test_second_allocation_failure_rolls_back_first(self):
        native = FakeNative()
        native.fail_create = 1
        with self.assertRaisesRegex(RuntimeError, "create failure"):
            self.load(native)
        self.assertEqual(native.close_attempts, [0])

    def test_read_or_load_failure_rolls_back_without_export(self):
        for exception_type in (RuntimeError, KeyboardInterrupt):
            native = FakeNative()
            native.fail_load, native.load_error = 2, exception_type
            with self.assertRaises(exception_type):
                self.load(native)
            self.assertEqual(native.close_attempts, [0])
            self.assertFalse(any(event[0] == "export" for event in native.events))

    def test_file_open_failure_after_preflight_rolls_back_first_owner(self):
        plan = self.inspect()
        native = FakeNative()
        with patch.object(m, "_open_tensor_source", side_effect=RuntimeError("injected file read failure")):
            with self.assertRaisesRegex(RuntimeError, "file read failure"):
                self.load(native, plan)
        self.assertEqual(native.close_attempts, [0])
        self.assertFalse(any(event[0] == "export" for event in native.events))

    def test_second_export_failure_reverses_all_owners_without_publication(self):
        native = FakeNative()
        native.fail_export = 1
        with self.assertRaisesRegex(RuntimeError, "export failure"):
            self.load(native)
        self.assertEqual(native.close_attempts, [1, 0])
        self.assertTrue(all(owner.closed for owner in native.owners))

    def test_cleanup_failures_retain_every_owner_and_can_retry(self):
        native = FakeNative()
        native.fail_export, native.fail_close = 1, {0, 1}
        with self.assertRaises(m.CleanupPendingError) as error:
            self.load(native)
        pending = error.exception.resource
        self.assertEqual(native.close_attempts, [1, 0])
        self.assertEqual(len(pending._owners), 2)
        self.assert_code("GENERATION_NOT_READY", lambda: pending.private_handles)
        self.assertNotIn("private-capability", repr(pending))
        native.fail_close.clear()
        pending.close()
        self.assertTrue(all(owner.closed for owner in native.owners))


if __name__ == "__main__":
    unittest.main()
