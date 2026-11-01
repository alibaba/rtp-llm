"""CPU-only staging tests: byte fidelity, safety, interruptions, bounded reads."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("stage_checkpoint_under_test", Path(__file__).with_name("stage_checkpoint.py"))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def write_shard(path, tensors):
    header = {"__metadata__": {"format": "pt", "custom": "preserve me"}}
    body = bytearray()
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(body), len(body) + len(raw)]}
        body.extend(raw)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + body)


def read_shard(path):
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    tensors = {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        a, b = item["data_offsets"]
        tensors[name] = (item["dtype"], item["shape"], raw[8 + length + a:8 + length + b])
    return header.get("__metadata__"), tensors


class StagingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        # /var is a symlink on macOS: fixture inputs use canonical real paths.
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.destination = self.root / "consumer"
        self.full = "model-00001-of-00048.safetensors"
        self.mixed1 = "model-00047-of-00048.safetensors"
        self.mixed2 = "model-00048-of-00048.safetensors"
        self.tensor_map = {
            self.full: [("layers.0.attention.weight", "BF16", [2, 2], bytes(range(8)))],
            self.mixed1: [
                ("layers.1.engram.embed.weight", "F8_E4M3", [4, 4], bytes(range(16))),
                ("layers.1.engram.embed.scale", "F8_E8M0", [4, 1], bytes([127] * 4)),
                ("layers.1.engram.q_weight", "BF16", [2, 2], b"\x01\x3f" * 4),
                ("layers.1.engram.wkv.scale", "F8_E8M0", [2, 2], b"\x7f" * 4),
            ],
            self.mixed2: [
                ("layers.14.engram.embed.weight", "F8_E4M3", [4, 4], bytes(range(16, 32))),
                ("layers.14.engram.embed.scale", "F8_E8M0", [4, 1], bytes([126] * 4)),
                ("layers.14.engram.k_weight", "BF16", [2, 2], b"\x02\x3f" * 4),
            ],
        }
        weight_map = {}
        size = 0
        for name, tensors in self.tensor_map.items():
            write_shard(self.source / name, tensors)
            weight_map.update({tensor[0]: name for tensor in tensors})
            size += sum(len(tensor[3]) for tensor in tensors)
        (self.source / stage.INDEX).write_text(json.dumps({"metadata": {"total_size": size, "revision": "fixture"}, "weight_map": weight_map}))
        (self.source / "config.json").write_text('{"model_type":"deepseek_v41"}')
        (self.source / "tokenizer.json").write_text('{"version":"1.0"}')
        (self.source / "encoding.py").write_text('# source revision stays unchanged\n')
        self.source_original = {p.name: p.read_bytes() for p in self.source.iterdir()}
        self.expected = {name: hashlib.sha256(raw).hexdigest() for name, raw in self.source_original.items()}

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, mode="consumer", expected=None):
        return stage.make_plan(self.source, self.destination, mode, self.expected if expected is None else expected)

    def execute(self, plan=None, **kwargs):
        return stage.execute_plan(plan or self.plan(), buffer_size=7, reserve_bytes=0, **kwargs)

    def assert_source_unchanged(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, self.source_original)

    def test_plan_is_read_only_and_accounts_exact_destination_size(self):
        plan = self.plan()
        self.assertFalse(self.destination.exists())
        self.assertFalse(Path(plan["stage"]).exists())
        self.assertEqual(plan["output_tensor_count"], 4)
        self.assertEqual(plan["output_tensor_bytes"], 28)
        self.assertEqual([x["kind"] for x in plan["outputs"][:3]], ["copy", "subset", "subset"])
        self.assert_source_unchanged()

    def test_consumer_preserves_raw_dtype_shape_and_nonembed_values(self):
        plan = self.plan()
        provenance = self.execute(plan)
        index = json.loads((self.destination / stage.INDEX).read_text())
        self.assertEqual(index["metadata"]["total_size"], 28)
        self.assertEqual(index["metadata"]["revision"], "fixture")
        self.assertEqual(len(index["weight_map"]), 4)
        for shard in self.tensor_map:
            source_meta, source_tensors = read_shard(self.source / shard)
            metadata, actual = read_shard(self.destination / shard)
            self.assertEqual(metadata, source_meta)
            self.assertEqual(actual, {key: value for key, value in source_tensors.items() if ".embed." not in key})
            self.assertFalse(any(".embed." in key for key in actual))
        self.assertEqual((self.destination / self.full).read_bytes(), self.source_original[self.full])
        for name in ("config.json", "tokenizer.json", "encoding.py"):
            self.assertEqual((self.destination / name).read_bytes(), self.source_original[name])
        for name, result in provenance["outputs"].items():
            self.assertEqual(result["sha256"], hashlib.sha256((self.destination / name).read_bytes()).hexdigest())
        self.assertIn("full source shard was not hashed", provenance["outputs"][self.mixed1]["source_digest_scope"])
        self.assertEqual(sum(x["bytes"] for x in provenance["outputs"].values()), plan["output_bytes"])
        self.assert_source_unchanged()

    def test_owner_only_copies_shards_with_tables_and_rewrites_index(self):
        plan = self.plan("owner")
        self.execute(plan)
        self.assertFalse((self.destination / self.full).exists())
        for name in (self.mixed1, self.mixed2):
            self.assertEqual((self.destination / name).read_bytes(), self.source_original[name])
        index = json.loads((self.destination / stage.INDEX).read_text())
        self.assertEqual(set(index["weight_map"].values()), {self.mixed1, self.mixed2})
        self.assertEqual(len(index["weight_map"]), 7)
        self.assert_source_unchanged()

    def test_streamed_expected_hash_mismatch_never_publishes(self):
        expected = dict(self.expected, **{self.full: "0" * 64})
        with self.assertRaisesRegex(stage.StageError, "streamed SHA256"):
            self.execute(self.plan(expected=expected))
        self.assertFalse(self.destination.exists())

    def test_expected_metadata_and_index_mismatch_are_preflight_errors(self):
        for name in ("config.json", stage.INDEX):
            with self.subTest(name=name), self.assertRaisesRegex(stage.StageError, "SHA256"):
                self.plan(expected=dict(self.expected, **{name: "0" * 64}))

    def test_disk_preflight_does_not_create_stage(self):
        plan = self.plan()
        usage = shutil.disk_usage(str(self.root))
        with mock.patch.object(stage.shutil, "disk_usage", return_value=type(usage)(usage.total, usage.total, 0)):
            with self.assertRaisesRegex(stage.StageError, "insufficient"):
                self.execute(plan)
        self.assertFalse(Path(plan["stage"]).exists())

    def test_source_changed_after_plan_is_rejected(self):
        plan = self.plan()
        with (self.source / self.full).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(stage.StageError, "source changed"):
            self.execute(plan)
        self.assertFalse(self.destination.exists())

    def test_source_mutation_during_copy_is_rejected(self):
        plan, original = self.plan(), stage._copy_range
        def mutate(*args):
            original(*args)
            with (self.source / self.full).open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\x99")
        with mock.patch.object(stage, "_copy_range", side_effect=mutate):
            with self.assertRaisesRegex(stage.StageError, "source changed"):
                self.execute(plan)
        self.assertFalse(self.destination.exists())

    def interrupt_after_first_file(self, plan):
        def stop(event, name, size):
            if event == "complete":
                raise InterruptedError("test interrupted after durable file")
        with self.assertRaises(InterruptedError):
            self.execute(plan, progress=stop)

    def test_resume_rehashes_completed_file_and_finishes(self):
        plan = self.plan()
        self.interrupt_after_first_file(plan)
        calls = []
        original = stage._hash_file
        def hash_spy(path, size):
            calls.append(path.name)
            return original(path, size)
        with mock.patch.object(stage, "_hash_file", side_effect=hash_spy):
            self.execute(self.plan(), resume=True)
        self.assertIn(self.full, calls)
        self.assertTrue(self.destination.is_dir())
        self.assert_source_unchanged()

    def test_resume_rejects_corrupted_complete_file(self):
        plan = self.plan()
        self.interrupt_after_first_file(plan)
        path = Path(plan["stage"]) / self.full
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        with self.assertRaisesRegex(stage.StageError, "resume output SHA256"):
            self.execute(plan, resume=True)

    def test_resume_discards_only_incomplete_part_file(self):
        plan = self.plan()
        self.interrupt_after_first_file(plan)
        (Path(plan["stage"]) / (self.mixed1 + ".part")).write_bytes(b"unfinished")
        self.execute(plan, resume=True)
        self.assertTrue(self.destination.is_dir())

    def test_resume_recovers_crash_before_or_after_file_rename(self):
        original = stage._rename_noreplace
        for after_rename in (False, True):
            with self.subTest(after_rename=after_rename):
                self.destination = self.root / ("recovered-" + str(after_rename))
                plan = self.plan()
                def interrupt(source, destination):
                    if str(source).endswith(self.full + ".part"):
                        if after_rename:
                            original(source, destination)
                        raise InterruptedError("file-commit crash")
                    return original(source, destination)
                with mock.patch.object(stage, "_rename_noreplace", side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        self.execute(plan)
                self.execute(plan, resume=True)
                self.assertTrue(self.destination.is_dir())

    def test_resume_refuses_changed_source_and_unknown_staging_files(self):
        plan = self.plan()
        self.interrupt_after_first_file(plan)
        (Path(plan["stage"]) / "unrelated").write_text("do not delete")
        with self.assertRaisesRegex(stage.StageError, "untracked file"):
            self.execute(plan, resume=True)
        self.assertTrue((Path(plan["stage"]) / "unrelated").exists())

    def test_existing_destination_is_not_overwritten(self):
        self.destination.mkdir()
        (self.destination / "keep").write_text("keep")
        with self.assertRaisesRegex(stage.StageError, "destination already exists"):
            self.plan()
        self.assertEqual((self.destination / "keep").read_text(), "keep")

    def test_atomic_rename_never_replaces_empty_competing_directory(self):
        a, b = self.root / "a", self.root / "b"
        a.mkdir(); b.mkdir()
        with self.assertRaises(OSError):
            stage._rename_noreplace(a, b)
        self.assertTrue(a.exists() and b.exists())

    def test_rejects_symlink_source_target_and_metadata(self):
        linked = self.root / "link"
        linked.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(stage.StageError, "symlink"):
            stage.make_plan(linked, self.destination, "consumer")
        target_link = self.root / "targetlink"
        target_link.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(stage.StageError, "symlink"):
            stage.make_plan(self.source, target_link, "consumer")
        (self.source / "config.json").unlink()
        (self.source / "config.json").symlink_to(self.root / "outside")
        with self.assertRaisesRegex(stage.StageError, "symlink"):
            self.plan()

    def test_rejects_source_inside_destination_or_destination_inside_source(self):
        with self.assertRaisesRegex(stage.StageError, "contain one another"):
            stage.make_plan(self.source, self.source / "new", "consumer")

    def test_rejects_index_path_traversal_and_duplicate_keys(self):
        index = self.source / stage.INDEX
        index.write_text('{"weight_map":{"x":"../outside.safetensors"}}')
        with self.assertRaisesRegex(stage.StageError, "plain filenames"):
            self.plan(expected={})
        index.write_text('{"weight_map":{},"weight_map":{}}')
        with self.assertRaisesRegex(stage.StageError, "duplicate"):
            self.plan(expected={})

    def test_rejects_malformed_header_and_index_header_mismatch(self):
        path = self.source / self.full
        original = path.read_bytes()
        path.write_bytes(struct.pack("<Q", stage.MAX_HEADER + 1) + original[8:])
        with self.assertRaisesRegex(stage.StageError, "oversized"):
            self.plan()
        write_shard(path, [("unlisted", "U8", [1], b"\x00")])
        with self.assertRaisesRegex(stage.StageError, "index/header"):
            self.plan()

    def test_rejects_incomplete_embedding_pairs(self):
        index_path = self.source / stage.INDEX
        index = json.loads(index_path.read_text())
        del index["weight_map"]["layers.1.engram.embed.scale"]
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(stage.StageError, "pair is incomplete"):
            self.plan(expected={})

    def test_consumer_never_reads_embedding_payload_ranges(self):
        plan = self.plan()
        reads = []
        original = stage._copy_range
        def spy(source, output, count, *rest):
            source_inode = os.fstat(source.fileno()).st_ino
            for filename in (self.mixed1, self.mixed2):
                if source_inode == (self.source / filename).stat().st_ino:
                    reads.append((filename, source.tell(), count))
            return original(source, output, count, *rest)
        with mock.patch.object(stage, "_copy_range", side_effect=spy):
            self.execute(plan)
        expected = []
        for item in plan["outputs"]:
            if item["kind"] == "subset":
                expected.extend((item["name"], item["source_data_start"] + x["offset"], x["bytes"]) for x in item["ranges"])
        self.assertEqual(reads, expected)

    def test_copy_reads_are_bounded(self):
        class Source:
            def __init__(self):
                self.sizes = []
            def read(self, size):
                self.sizes.append(size)
                return b"x" * size
        class Sink:
            def write(self, raw):
                pass
        source = Source()
        stage._copy_range(source, Sink(), 103, 7, hashlib.sha256(), None)
        self.assertEqual(sum(source.sizes), 103)
        self.assertLessEqual(max(source.sizes), 7)

    def test_no_second_full_read_after_new_copy(self):
        with mock.patch.object(stage, "_hash_file", side_effect=AssertionError("new output must not be re-read")):
            self.execute()

    def test_nested_encoding_metadata_is_copied_and_resumable(self):
        (self.source / "encoding").mkdir()
        raw = b"# preserve the exact source encoder revision\n"
        (self.source / "encoding/encoding.py").write_bytes(raw)
        plan = self.plan()
        def stop(event, name, size):
            if event == "complete" and name == "encoding/encoding.py":
                raise InterruptedError("after nested metadata")
        with self.assertRaises(InterruptedError):
            self.execute(plan, progress=stop)
        provenance = self.execute(plan, resume=True)
        self.assertEqual((self.destination / "encoding/encoding.py").read_bytes(), raw)
        self.assertEqual(provenance["outputs"]["encoding/encoding.py"]["sha256"], hashlib.sha256(raw).hexdigest())

    def test_nested_metadata_rejects_symlink_directory_and_traversal(self):
        outside = self.root / "other"
        outside.mkdir()
        (outside / "encoding.py").write_text("must not read")
        (self.source / "encoding").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(stage.StageError, "symlink"):
            self.plan()
        with self.assertRaisesRegex(stage.StageError, "plain filenames"):
            stage._metadata_name("../outside")

    def test_consumer_seeks_beyond_4gib_without_reading_sparse_embedding(self):
        name = "large-offset.safetensors"
        length = 2 ** 32 + 257
        keys = ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale", "layers.1.engram.q_weight")
        header = {
            keys[0]: {"dtype": "F8_E4M3", "shape": [length], "data_offsets": [0, length]},
            keys[1]: {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [length, length + 1]},
            keys[2]: {"dtype": "U8", "shape": [4], "data_offsets": [length + 1, length + 5]},
        }
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        with (self.source / name).open("wb") as output:
            output.write(struct.pack("<Q", len(raw)) + raw)
            output.seek(8 + len(raw) + length)
            output.write(b"\x7fTEST")
        (self.source / stage.INDEX).write_text(json.dumps({"metadata": {"total_size": length + 5}, "weight_map": {k: name for k in keys}}))
        plan = self.plan(expected={})
        self.assertLess(plan["output_bytes"], 4096)
        self.execute(plan)
        _, tensors = read_shard(self.destination / name)
        self.assertEqual(tensors, {keys[2]: ("U8", [4], b"TEST")})


if __name__ == "__main__":
    unittest.main()
