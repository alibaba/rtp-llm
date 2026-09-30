"""Named Engram tensor reads, byte preservation and stage ownership admission."""

import dataclasses
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_engram_test import engram, host, linear
from rtp_llm.utils.safetensors_header import named_safetensors_spans

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_engram": engram,
        "rtp_llm.models_py.modules.dsv4.v41_engram_host": host,
        "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
    },
):
    loading = load_codec_source("_dsv41_engram_weights_test", "v41_engram_weights.py")


def write_checkpoint(root):
    config = SimpleNamespace(
        text=dict(
            num_hidden_layers=40,
            hidden_size=32,
            hc_mult=4,
            rms_norm_eps=1e-20,
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[7, 11],
            engram_n_heads=1,
            engram_head_dim=64,
            engram_max_ngram_size=4,
        )
    )
    tensors, kinds = {}, {}
    for layer, rows in ((1, 7), (14, 11)):
        prefix = f"layers.{layer}.engram."
        entries = {
            "embed.weight": (
                torch.full((rows, 64), 0.5, dtype=torch.float8_e4m3fn),
                "F8_E4M3",
            ),
            "embed.scale": (torch.full((rows, 2), 127, dtype=torch.uint8), "F8_E8M0"),
            "wkv.weight": (
                (torch.arange(160 * 192).reshape(160, 192) % 9 - 4).to(
                    torch.float8_e4m3fn
                ),
                "F8_E4M3",
            ),
            "wkv.scale": (
                (torch.arange(30).reshape(5, 6) % 5 + 119).to(torch.uint8),
                "F8_E8M0",
            ),
            "q_weight": (torch.linspace(-1, 1, 128).reshape(4, 32).bfloat16(), "BF16"),
            "k_weight": (torch.full((4, 32), -0.0, dtype=torch.bfloat16), "BF16"),
        }
        for suffix, (tensor, dtype) in entries.items():
            tensors[prefix + suffix], kinds[prefix + suffix] = tensor, dtype
    tensors["unrelated.weight"] = torch.full((1024 * 1024,), 0xA5, dtype=torch.uint8)
    kinds["unrelated.weight"] = "U8"
    header, payload = {}, bytearray()
    for name, tensor in tensors.items():
        raw = tensor.view(torch.uint8).numpy().tobytes()
        header[name] = dict(
            dtype=kinds[name],
            shape=list(tensor.shape),
            data_offsets=[len(payload), len(payload) + len(raw)],
        )
        payload.extend(raw)
    encoded = json.dumps(header).encode()
    (root / "weights.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )
    return config, tensors


class RecordingReader:
    def __init__(self, file, records, *, max_read=None, on_read=None):
        self.file, self.records = file, records
        self.max_read, self.on_read = max_read, on_read

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.file.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.file, name)

    def readinto(self, target):
        self.records.append((self.file.tell(), len(target)))
        count = self.file.readinto(
            target if self.max_read is None else target[: self.max_read]
        )
        if self.on_read is not None:
            self.on_read()
        return count


class V41EngramWeightsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config, self.tensors = write_checkpoint(self.root)
        self.sources = loading.plan_stage_engram_sources(
            self.config, self.root, range(21), "a" * 64
        )

    def owner(self, source):
        table = host._PinnedSharedTable(
            source.table, max_bytes=source.table.nbytes, directory=self.root
        )
        self.addCleanup(table.close)
        return host.HostEngramEmbedding(table)

    def test_stage_ownership_and_no_io_on_stage_one(self):
        self.assertEqual(set(self.sources), {1, 14})
        self.assertEqual(
            set(
                loading.plan_stage_engram_sources(
                    self.config, self.root, range(40), "a" * 64
                )
            ),
            {1, 14},
        )
        with patch.object(
            Path, "open", side_effect=AssertionError("stage-one checkpoint I/O")
        ):
            self.assertEqual(
                loading.plan_stage_engram_sources(
                    self.config, "/missing", range(21, 40), "a" * 64
                ),
                {},
            )
        for owned in ([1, 1], [-1], [40], [True]):
            with self.assertRaises(ValueError):
                loading.plan_stage_engram_sources(
                    self.config, self.root, owned, "a" * 64
                )

    def test_reads_exactly_four_named_spans_with_bitwise_dtype_preservation(self):
        source = self.sources[1]
        records, actual_open = [], open
        with patch(
            "builtins.open",
            side_effect=lambda *a, **kw: RecordingReader(
                actual_open(*a, **kw), records
            ),
        ):
            loaded = loading.load_engram_weights(source, max_bytes=source.weight_bytes)
        self.assertEqual(records, [(span.offset, span.size) for span in source.spans])
        for name, tensor in loaded.items():
            span = getattr(source, name)
            self.assertEqual(tensor.dtype, self.tensors[span.name].dtype)
            torch.testing.assert_close(
                tensor.view(torch.uint8),
                self.tensors[span.name].view(torch.uint8),
                rtol=0,
                atol=0,
            )
        self.assertTrue(bool(torch.signbit(loaded["k_weight"]).all()))
        self.assertEqual(source.packed_scale_bytes, 256 * 8)

    def test_partial_reads_are_completed_and_io_requests_are_bounded(self):
        span = self.sources[1].q_weight
        actual_open, records = open, []
        output = bytearray(span.size)
        with patch(
            "builtins.open",
            side_effect=lambda *a, **kw: RecordingReader(
                actual_open(*a, **kw), records, max_read=7
            ),
        ):
            span.read_into(output, chunk_bytes=17)
        self.assertLessEqual(max(length for _, length in records), 17)
        self.assertEqual(
            output, self.tensors[span.name].view(torch.uint8).numpy().tobytes()
        )
        with self.assertRaises(ValueError):
            span.read_into(bytearray(span.size - 1))
        with self.assertRaises(ValueError):
            span.read_into(bytes(span.size))

    def test_budgets_and_backend_are_rejected_before_tensor_reads(self):
        source = self.sources[1]
        embedding = self.owner(source)
        with patch.object(
            loading, "_load_tensor", side_effect=AssertionError("payload read")
        ):
            for budget in (source.weight_bytes - 1, True):
                with self.assertRaisesRegex(ValueError, "budget"):
                    loading.load_engram_weights(source, max_bytes=budget)
            with self.assertRaisesRegex(ValueError, "packed-scale"):
                loading.load_engram_module(
                    source, embedding, device="cuda", max_bytes=source.weight_bytes
                )
            with self.assertRaisesRegex(ValueError, "CPU-only"):
                loading.load_engram_module(
                    source, embedding, device="cpu", max_bytes=10**9
                )

    def test_changed_source_is_rejected_before_read(self):
        source = self.sources[1]
        path = Path(source.weight.path)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        with patch.object(
            loading, "_load_tensor", side_effect=AssertionError("payload read")
        ):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                loading.load_engram_weights(source, max_bytes=source.weight_bytes)

    def test_changes_during_read_and_unexpected_eof_are_rejected(self):
        span = self.sources[1].q_weight
        path = Path(span.path)
        stat = path.stat()
        actual_open, records = open, []

        def change():
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))

        with patch(
            "builtins.open",
            side_effect=lambda *a, **kw: RecordingReader(
                actual_open(*a, **kw), records, on_read=change
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "during tensor read"):
                span.read_into(bytearray(span.size))
        fresh = loading.plan_stage_engram_sources(
            self.config, self.root, [1], "a" * 64
        )[1].q_weight
        with patch.object(RecordingReader, "readinto", return_value=0), patch(
            "builtins.open",
            side_effect=lambda *a, **kw: RecordingReader(actual_open(*a, **kw), []),
        ):
            with self.assertRaisesRegex(ValueError, "ended"):
                fresh.read_into(bytearray(fresh.size))

    def test_wrong_shape_dtype_name_and_shard_paths_are_rejected(self):
        name = "layers.1.engram.q_weight"
        for expected in (
            {name: ((32, 4), "BF16")},
            {name: ((4, 32), "F16")},
            {"missing": ((1,), "U8")},
        ):
            with self.assertRaises((ValueError, KeyError)):
                named_safetensors_spans(self.root, expected)
        index = self.root / "model.safetensors.index.json"
        for filename in ("../weights.safetensors", "", "/tmp/weights.safetensors"):
            index.write_text(json.dumps({"weight_map": {name: filename}}))
            with self.assertRaisesRegex(ValueError, "checkpoint directory"):
                named_safetensors_spans(self.root, {name: ((4, 32), "BF16")})

    def test_external_shard_symlink_and_duplicate_index_keys_are_rejected(self):
        external = self.root.parent / (self.root.name + ".safetensors")
        external.write_bytes(b"outside")
        self.addCleanup(external.unlink)
        (self.root / "external.safetensors").symlink_to(external)
        name = "layers.1.engram.q_weight"
        index = self.root / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {name: "external.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "checkpoint directory"):
            named_safetensors_spans(self.root, {name: ((4, 32), "BF16")})
        index.write_text('{"weight_map": {}, "weight_map": {}}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            named_safetensors_spans(self.root, {name: ((4, 32), "BF16")})

    def test_loaded_module_matches_checkpoint_and_owner_identity(self):
        source = self.sources[1]
        embedding = self.owner(source)
        module = loading.load_engram_module(
            source,
            embedding,
            device="cpu",
            max_bytes=source.weight_bytes,
            chunk_tokens=2,
            backend="eager",
        )
        hidden = torch.ones(5, 4, 32, dtype=torch.bfloat16)
        ids = torch.zeros(5, 3, dtype=torch.int64)
        valid = torch.tensor([True, False, True, True, False])
        expected = engram.V41Engram(
            embedding,
            linear.V41MXFP8Linear(
                self.tensors[source.weight.name],
                self.tensors[source.scales.name],
                backend="eager",
            ),
            self.tensors[source.q_weight.name],
            self.tensors[source.k_weight.name],
            eps=1e-20,
        )(hidden, ids, valid)
        torch.testing.assert_close(module(hidden, ids, valid), expected, rtol=0, atol=0)
        other = dataclasses.replace(
            source, table=dataclasses.replace(source.table, checkpoint_sha256="b" * 64)
        )
        with self.assertRaisesRegex(ValueError, "one admitted source"):
            loading.load_engram_module(
                other, embedding, device="cpu", max_bytes=10**9, backend="eager"
            )
        embedding._table.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            loading.load_engram_module(
                source, embedding, device="cpu", max_bytes=10**9, backend="eager"
            )


if __name__ == "__main__":
    unittest.main()
