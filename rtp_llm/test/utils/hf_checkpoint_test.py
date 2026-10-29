import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.utils.hf_checkpoint import resolve_hf_checkpoint


class HFCheckpointTest(unittest.TestCase):
    URI = "hf://owner/model@" + "a" * 40

    def setUp(self):
        resolve_hf_checkpoint.cache_clear()
        self.hub = SimpleNamespace(
            hf_hub_download=Mock(),
            snapshot_download=Mock(return_value="/cache/snapshot"),
        )
        self.modules = patch.dict(sys.modules, {"huggingface_hub": self.hub})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.addCleanup(resolve_hf_checkpoint.cache_clear)

    def test_local_paths_do_not_download(self):
        self.assertEqual(resolve_hf_checkpoint("/mnt/model"), "/mnt/model")
        self.hub.snapshot_download.assert_not_called()

    def test_mutable_or_malformed_revision_is_rejected(self):
        for uri in (
            "hf://owner/model",
            "hf://owner/model@main",
            "hf://owner/model@aaaaaaa",
        ):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                resolve_hf_checkpoint(uri)
        self.hub.snapshot_download.assert_not_called()

    def test_full_snapshot_uses_pin_and_reuses_download_for_mtp(self):
        with patch.dict(os.environ, {"HF_HUB_CACHE": "/persistent/hf"}):
            self.assertEqual(resolve_hf_checkpoint(self.URI), "/cache/snapshot")
            self.assertEqual(resolve_hf_checkpoint(self.URI), "/cache/snapshot")
        self.hub.snapshot_download.assert_called_once_with(
            repo_id="owner/model",
            revision="a" * 40,
            cache_dir="/persistent/hf",
            token=False,
            allow_patterns=None,
            max_workers=4,
        )

    def test_partial_snapshot_keeps_all_matching_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "residual.a": "shard1.safetensors",
                            "residual.b": "shard2.safetensors",
                            "q_norm.weight": "shard2.safetensors",
                            "unrelated": "shard3.safetensors",
                        }
                    }
                )
            )
            self.hub.hf_hub_download.return_value = str(index)
            resolve_hf_checkpoint(self.URI, ("residual.*", "q_norm.weight"))
            self.assertEqual(
                self.hub.snapshot_download.call_args.kwargs["allow_patterns"],
                ["*.json", "shard1.safetensors", "shard2.safetensors"],
            )
            self.hub.snapshot_download.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "no tensors matching"):
                resolve_hf_checkpoint(self.URI, ("missing.*",))
            self.hub.snapshot_download.assert_not_called()

    def test_failed_download_cannot_become_a_local_fixture(self):
        self.hub.snapshot_download.side_effect = OSError("network unavailable")
        with self.assertRaisesRegex(OSError, "network unavailable"):
            resolve_hf_checkpoint(self.URI)


if __name__ == "__main__":
    unittest.main()
