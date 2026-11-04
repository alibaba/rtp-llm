# SPDX-License-Identifier: Apache-2.0
"""Immutable metadata and real raw-row loading tests; no CUDA allocation.

Torch/safetensors are required for CheckpointTest; the reader boundary tests
also run on a dependency-free CPU interpreter.
"""
import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rtp_llm.models_py.modules.dsv41 import engram_checkpoint as m
from rtp_llm.models_py.modules.dsv41.engram_publish import load_rows

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

    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(m.CheckpointError) as error:
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
        wrong = m.ExpectedLayout(layers=((1, 6), (2, 131)), head_dim=64,
                                 model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi)
        self.assert_code("SHAPE_MISMATCH", m.inspect_checkpoint, self.root, wrong)

    def test_lookup_width_and_manifest_tensor_count_fail_in_preflight(self):
        for layout, code in (
                (m.ExpectedLayout(layers=self.layout.layers, head_dim=96,
                                  model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi),
                 "UNSUPPORTED_LOOKUP_WIDTH"),
                (m.ExpectedLayout(layers=tuple((i, 1) for i in range(513)), head_dim=64,
                                  model_revision=self.layout.model_revision, hash_abi=self.layout.hash_abi),
                 "MANIFEST_LIMIT")):
            self.assert_code(code, m.inspect_checkpoint, self.root, layout)

    def inspect(self):
        return m.inspect_checkpoint(self.root, self.layout)

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
                    checksum = load_rows(Owner(), self.root, tensor,
                        snapshots[tensor.relative_path], start, end, width,
                        chunk_bytes=2 * width)
                    expected = self.payloads[tensor.name][start * width:end * width]
                    self.assertEqual(b''.join(data for _, data in calls), expected)
                    self.assertEqual(checksum, 'sha256:' + hashlib.sha256(expected).hexdigest())
                    self.assertTrue(all(len(data) <= 2 * width for _, data in calls))
                    self.assertEqual([offset for offset, _ in calls],
                                     list(range(0, len(expected), 2 * width)))

    def test_source_replacement_after_inspection_fails_before_native_write(self):
        plan = self.inspect()
        tensor = plan.layers[0].tensors[0]
        snapshot = next(s for s in plan.files if s.relative_path == tensor.relative_path)
        path = self.root / tensor.relative_path
        replacement = self.root / 'replacement'
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        owner = unittest.mock.Mock()
        with self.assertRaisesRegex(m.CheckpointError, 'CHECKPOINT_CHANGED'):
            load_rows(owner, self.root, tensor, snapshot, 0, 1, tensor.shape[1])
        owner.load.assert_not_called()

    def test_duplicate_header_or_truncated_payload_is_rejected(self):
        for duplicate in (True, False):
            self.write_checkpoint()
            path = self.root / 'weights.safetensors'
            raw = path.read_bytes()
            if duplicate:
                length = int.from_bytes(raw[:8], 'little')
                header = b'{"__metadata__":{},"__metadata__":{},' + raw[8:8 + length].lstrip()[1:]
                header += b' ' * (-len(header) % 8)
                path.write_bytes(len(header).to_bytes(8, 'little') + header + raw[8 + length:])
            else:
                path.write_bytes(raw[:-1])
            with self.assertRaises(Exception):
                self.inspect()
    def test_self_publish_loads_only_each_shard_and_rejects_changed_index(self):
        import sys
        from contextlib import nullcontext
        from types import SimpleNamespace as NS
        from uuid import uuid4
        from rtp_llm.models_py.modules.dsv41 import engram_bootstrap as bootstrap
        from rtp_llm.models_py.modules.dsv41 import engram_publish as publisher
        from rtp_llm.models_py.modules.dsv41.engram_descriptors import validate_documents

        self.layout = m.ExpectedLayout(layers=((1, 65), (2, 131)), head_dim=64,
            model_revision='real-publish-test', hash_abi='sha256:' + 'a' * 64)
        self.write_checkpoint()
        (self.root / 'config.json').write_text('{}')
        layout = NS(layer_ids=(1, 2), num_embeddings=(65, 131), head_dim=64)
        model = NS(EngramLayout=lambda _: layout, NgramHashState=lambda *a: None)
        created = []
        mutate = [False]
        class Owner:
            def __init__(self, size, *args):
                self.data = bytearray(size)
                self.closed = False
                self.handle = uuid4().bytes
                created.append(self)
            def load(self, raw, offset):
                self.data[offset:offset+len(raw)] = raw
                if mutate[0]:
                    mutate[0] = False
                    path = self_root / 'model.safetensors.index.json'
                    path.write_text(path.read_text() + ' ')
            def export_handle(self): return self.handle
            def close(self): self.closed = True
        self_root = self.root
        native = NS(Owner=Owner, host_numa_granularity=lambda *a: 64,
                    max_allocation_bytes=1 << 30, max_load_chunk_bytes=128)
        generation = str(uuid4())
        with patch.dict(sys.modules, {'rtp_llm.models_py.modules.dsv41.engram': model}), \
             patch.object(bootstrap, '_load_native', lambda: native), \
             patch.object(bootstrap, 'hash_state_digest', lambda _: self.layout.hash_abi), \
             patch.object(torch.cuda, 'device', lambda _: nullcontext()):
            documents = [publisher.publish_shard(self.root, self.layout.model_revision,
                         f'host-{part}', generation, part, 2) for part in range(2)]
            validate_documents(documents, 2, generation)
            for document in documents:
                for name, tensor in document['tensors'].items():
                    raw = self.payloads[name]
                    width = tensor['shape'][1]
                    expected = raw[tensor['row_begin']*width:tensor['row_end']*width]
                    self.assertEqual(tensor['sha256'], 'sha256:' + hashlib.sha256(expected).hexdigest())
            self.assertEqual(publisher.release(generation), [])
            self.assertTrue(all(owner.closed for owner in created))
            created.clear()
            mutate[0] = True
            with self.assertRaisesRegex(m.CheckpointError, 'CHECKPOINT_CHANGED'):
                publisher.publish_shard(self.root, self.layout.model_revision, 'host-0', generation, 0, 2)
            self.assertTrue(created and all(owner.closed for owner in created))
            self.assertNotIn(generation, publisher._KEEPERS)


if __name__ == '__main__':
    unittest.main()
