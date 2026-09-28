"""Checkpoint source checks for the K3 example launcher."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


LAUNCHER = Path(__file__).with_name("launch_bf16.py")
spec = importlib.util.spec_from_file_location("k3_launch_bf16", LAUNCHER)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class CheckpointSourceTest(unittest.TestCase):
    def test_direct_3fs_launcher_selects_supported_fastsafetensors_copier(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            (target / "config.json").write_text(json.dumps({"num_hidden_layers": 93}))
            args = SimpleNamespace(
                checkpoint=str(target), draft_checkpoint=str(draft),
                start_port=28000, peer_port=29000, peer_ip="127.0.0.1",
                role="PREFILL", server="/bin/true", allow_hf3fs_root=str(root),
            )
            environment, _ = launch.launch_config(args)
            self.assertEqual(environment["FASTSAFETENSORS_NOGDS"], "0")

    def test_explicit_hf3fs_root_accepts_only_matching_fuse_mount(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp) / "weights"
            model = root / "target"
            other = Path(tmp) / "other"
            model.mkdir(parents=True)
            other.mkdir()
            with patch.object(launch.subprocess, "check_output", return_value="fuse.hf3fs\nfuse.hf3fs\n"):
                launch.require_checkpoint_source(model, root)
                with self.assertRaises(ValueError):
                    launch.require_checkpoint_source(other, root)
            with patch.object(launch.subprocess, "check_output", return_value="ext4\n"):
                with self.assertRaises(ValueError):
                    launch.require_checkpoint_source(model, root)

    def test_network_checkpoint_remains_rejected_without_explicit_root(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            with self.assertRaises(ValueError):
                launch.require_checkpoint_source(Path(tmp), None)


if __name__ == "__main__":
    unittest.main()
