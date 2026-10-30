"""Checkpoint corruption and Bazel-native storage boundary regressions."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.smoke.kimi_k3 import checkpoint_guard as guard
from rtp_llm.test.smoke.kimi_k3 import server_launcher as launch


class CheckpointGuardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text('{"num_hidden_layers":93}')
        self.mapping = {
            "layer.0.weight": "model-01.safetensors",
            "layer.1.weight": "model-02.safetensors",
        }
        for name, relative in self.mapping.items():
            self.shard(
                relative,
                {name: {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}},
                b"\0" * 4,
            )
        self.index()

    def shard(self, relative, header, payload):
        encoded = json.dumps(header).encode() if isinstance(header, dict) else header
        path = self.checkpoint / relative
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path

    def index(self, total_size=8):
        (self.checkpoint / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": self.mapping}
            )
        )

    def check(self, **kwargs):
        return guard.preflight(
            self.checkpoint, allow_checkpoint_root=self.root, **kwargs
        )

    def test_complete_checkpoint_and_exact_shard_pin(self):
        result = self.check(expected_shards=2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["shard_count"], 2)
        self.assertEqual(result["tensor_count"], 2)
        self.assertEqual(result["tensor_bytes"], 8)
        self.assertEqual(result["index_total_size_accounting"], "tensor-data-bytes")
        self.assertFalse(result["weight_payload_checksums_verified"])
        self.assertEqual(len(result["index_sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "Expected 9 shards"):
            self.check(expected_shards=9)

    def test_shared_checkpoint_metadata_does_not_read_shard_headers(self):
        with patch.object(
            guard, "read_shard", side_effect=AssertionError("layout scan")
        ):
            result = self.check(check_layout=False, expected_shards=2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["validation"], "index-and-files")
        self.assertIsNone(result["tensor_bytes"])
        (self.checkpoint / "model-01.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            self.check(check_layout=False)

    def test_shared_checkpoint_metadata_retains_pin_and_path_checks(self):
        with self.assertRaisesRegex(ValueError, "Expected 3 shards"):
            self.check(check_layout=False, expected_shards=3)
        self.mapping["layer.0.weight"] = "../outside.safetensors"
        self.index()
        with self.assertRaisesRegex(ValueError, "Unsafe checkpoint file name"):
            self.check(check_layout=False)

    def test_cli_preserves_preflight_contract(self):
        result = subprocess.run(
            [
                sys.executable,
                guard.__file__,
                "preflight",
                "--checkpoint",
                str(self.checkpoint),
                "--load-method",
                "fastsafetensors",
                "--allow-checkpoint-root",
                str(self.root),
                "--expected-shards",
                "2",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout)["tensor_count"], 2)

    def test_missing_shard_fails_before_success(self):
        (self.checkpoint / "model-02.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_shared_shard_is_resolved_once_for_all_indexed_tensors(self):
        self.mapping["layer.1.weight"] = "model-01.safetensors"
        self.shard(
            "model-01.safetensors",
            {
                "layer.0.weight": {
                    "dtype": "BF16",
                    "shape": [2],
                    "data_offsets": [0, 4],
                },
                "layer.1.weight": {
                    "dtype": "BF16",
                    "shape": [2],
                    "data_offsets": [4, 8],
                },
            },
            b"\0" * 8,
        )
        self.index()
        with patch.object(guard, "_file", wraps=guard._file) as resolve:
            self.assertEqual(self.check()["tensor_count"], 2)
        shard_names = [
            call.args[2]
            for call in resolve.call_args_list
            if call.args[2].endswith(".safetensors")
        ]
        self.assertEqual(shard_names, ["model-01.safetensors"])

    def test_bad_header_length_and_truncated_payload_fail(self):
        path = self.checkpoint / "model-01.safetensors"
        original = path.read_bytes()
        for data in (
            b"short",
            struct.pack("<Q", guard.MAX_JSON_BYTES + 1),
            original[:-1],
        ):
            with self.subTest(data_length=len(data)):
                path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "header length|data offsets"):
                    self.check()

    def test_duplicate_json_keys_fail_in_index_and_header(self):
        index = self.checkpoint / "model.safetensors.index.json"
        original = index.read_bytes()
        index.write_text(
            '{"weight_map":{"w":"model-01.safetensors","w":"model-02.safetensors"}}'
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.check()
        index.write_bytes(original)
        raw = b'{"layer.0.weight":{"dtype":"BF16","dtype":"F16","shape":[2],"data_offsets":[0,4]}}'
        self.shard("model-01.safetensors", raw, b"\0" * 4)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.check()

    def test_shape_dtype_and_offsets_are_validated(self):
        invalid = (
            {"dtype": "UNSUPPORTED", "shape": [2], "data_offsets": [0, 4]},
            {"dtype": "BF16", "shape": [-2], "data_offsets": [0, 4]},
            {"dtype": "BF16", "shape": [True], "data_offsets": [0, 4]},
            {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]},
            {"dtype": "BF16", "shape": [2], "data_offsets": [4, 0]},
            {"dtype": "BF16", "shape": [2], "data_offsets": [False, 4]},
            {"dtype": "BF16", "shape": [2], "data_offsets": [-1, 3]},
        )
        for tensor in invalid:
            with self.subTest(tensor=tensor):
                self.shard(
                    "model-01.safetensors", {"layer.0.weight": tensor}, b"\0" * 4
                )
                with self.assertRaises(ValueError):
                    self.check()

    def test_overlap_gap_and_extra_payload_bytes_fail(self):
        for spans, length in (
            ([[0, 4], [2, 6]], 6),
            ([[0, 4], [5, 9]], 9),
            ([[0, 4]], 5),
        ):
            with self.subTest(spans=spans):
                header = {
                    f"weight.{i}": {
                        "dtype": "BF16",
                        "shape": [2],
                        "data_offsets": offsets,
                    }
                    for i, offsets in enumerate(spans)
                }
                self.shard("model-01.safetensors", header, b"\0" * length)
                with self.assertRaisesRegex(
                    ValueError, "Overlapping or gapped|Unindexed"
                ):
                    self.check()

    def test_scalar_empty_tensor_and_metadata_are_valid(self):
        header = {
            "scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]},
            "empty": {"dtype": "BF16", "shape": [0, 7], "data_offsets": [4, 4]},
            "__metadata__": {"format": "pt"},
        }
        path = self.shard("scalar.safetensors", header, b"\0" * 4)
        names, result = guard.read_shard(path)
        self.assertEqual(names, {"scalar", "empty"})
        self.assertEqual(result["tensor_bytes"], 4)

    def test_index_must_match_every_tensor_in_each_shard(self):
        self.mapping["other.weight"] = self.mapping.pop("layer.0.weight")
        self.index()
        with self.assertRaisesRegex(ValueError, "tensor membership mismatch"):
            self.check()

    def test_unreferenced_extra_tensor_in_shard_fails(self):
        self.shard(
            "model-01.safetensors",
            {
                "layer.0.weight": {
                    "dtype": "BF16",
                    "shape": [2],
                    "data_offsets": [0, 4],
                },
                "extra.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]},
            },
            b"\0" * 8,
        )
        with self.assertRaisesRegex(ValueError, "tensor membership mismatch"):
            self.check()

    def test_native_mtp_index_can_count_complete_shard_files(self):
        file_bytes = sum(
            (self.checkpoint / name).stat().st_size for name in self.mapping.values()
        )
        self.index(total_size=file_bytes)
        result = self.check()
        self.assertEqual(result["tensor_bytes"], 8)
        self.assertEqual(result["file_bytes"], file_bytes)
        self.assertEqual(result["index_total_size_accounting"], "shard-file-bytes")

    def test_index_total_size_must_match_one_exact_accounting(self):
        file_bytes = sum(
            (self.checkpoint / name).stat().st_size for name in self.mapping.values()
        )
        for value in (9, file_bytes - 1, file_bytes + 1, True, str(file_bytes)):
            with self.subTest(total_size=value):
                self.index(total_size=value)
                with self.assertRaisesRegex(ValueError, "total_size"):
                    self.check()

    def test_traversal_and_shard_symlink_escape_are_rejected(self):
        self.mapping["layer.0.weight"] = "../outside.safetensors"
        self.index()
        with self.assertRaisesRegex(ValueError, "Unsafe checkpoint file"):
            self.check()
        self.mapping["layer.0.weight"] = "model-01.safetensors"
        self.index()
        path = self.checkpoint / "model-01.safetensors"
        outside = self.root / "outside.safetensors"
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes approved directory"):
            self.check()

    def test_config_symlink_escape_and_source_root_escape_are_rejected(self):
        path = self.checkpoint / "config.json"
        outside = self.root / "outside.json"
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes approved directory"):
            self.check()
        with self.assertRaisesRegex(ValueError, "specific directory"):
            guard.preflight(self.checkpoint, allow_checkpoint_root="/")
        other = self.root / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "escapes approved storage root"):
            guard.preflight(self.checkpoint, allow_checkpoint_root=other)

    def test_storage_authorization_and_loader_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            guard.preflight(self.checkpoint)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.check(local_data_root=self.root)
        with self.assertRaisesRegex(ValueError, "fastsafetensors"):
            self.check(load_method="torch")
        with patch.object(guard, "_filesystem_type", return_value="ext4"):
            with self.assertRaisesRegex(ValueError, "Not a local data path"):
                guard.preflight(self.checkpoint, local_data_root=self.root)
        with patch.object(guard, "_filesystem_type", return_value="ext4"):
            with self.assertRaisesRegex(ValueError, "Expected fuse.hf3fs"):
                guard.preflight(self.checkpoint, allow_hf3fs_root=self.root)
        with patch.object(guard, "_filesystem_type", return_value="fuse.hf3fs"):
            self.assertTrue(
                guard.preflight(self.checkpoint, allow_hf3fs_root=self.root)["passed"]
            )


class BazelLauncherStorageTest(unittest.TestCase):
    def test_only_current_bazel_owned_output_and_tmp_paths_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output, tmp, foreign = root / "outputs", root / "tmp", root / "foreign"
            for path in (output, tmp, foreign):
                path.mkdir(mode=0o700)
            child = output / "task"
            child.mkdir(mode=0o700)
            with patch.dict(
                os.environ,
                {"TEST_UNDECLARED_OUTPUTS_DIR": str(output), "TEST_TMPDIR": str(tmp)},
                clear=True,
            ):
                launch.require_bazel_destination(child)
                launch.require_bazel_destination(tmp)
                with self.assertRaisesRegex(ValueError, "outside this test"):
                    launch.require_bazel_destination(foreign)
                child.chmod(0o777)
                with self.assertRaisesRegex(ValueError, "owned and private"):
                    launch.require_bazel_destination(child)
                child.chmod(0o700)
                with patch.object(
                    launch.os, "geteuid", return_value=child.stat().st_uid + 1
                ):
                    with self.assertRaisesRegex(ValueError, "owned and private"):
                        launch.require_bazel_destination(child)

    def test_bazel_output_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output, foreign = root / "outputs", root / "foreign"
            output.mkdir()
            foreign.mkdir()
            (foreign / "child").mkdir()
            (output / "link").symlink_to(foreign, target_is_directory=True)
            with patch.dict(
                os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": str(output)}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "outside this test"):
                    launch.require_bazel_destination(output / "link" / "child")
                with self.assertRaisesRegex(ValueError, "absolute owned"):
                    launch.require_bazel_destination(output / "link")

    def test_short_worker_tmp_is_scoped_to_uid_and_private_directory(self):
        with tempfile.TemporaryDirectory(
            prefix=f"k3tmp-{os.geteuid()}-", dir="/tmp"
        ) as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {}, clear=True):
                launch.require_bazel_temporary(root, "k3tmp")
                with self.assertRaisesRegex(ValueError, "outside this test"):
                    launch.require_bazel_temporary(root, "k3pd")
                root.chmod(0o755)
                with self.assertRaisesRegex(ValueError, "owned private"):
                    launch.require_bazel_temporary(root, "k3tmp")

    def test_bazel_native_rejects_four_layer_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, draft = root / "target", root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text('{"num_hidden_layers":4}')
            args = SimpleNamespace(
                checkpoint=target,
                draft_checkpoint=draft,
            )
            with self.assertRaisesRegex(ValueError, "complete 93-layer"):
                launch.launch_config(args)


if __name__ == "__main__":
    unittest.main()
