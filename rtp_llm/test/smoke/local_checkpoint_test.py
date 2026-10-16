import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.smoke.local_checkpoint import (
    relocate_model_paths,
    resolve_local_checkpoint,
)


class LocalCheckpointTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "DeepSeek-V4.1-Flash"
        self.checkpoint.mkdir()
        for filename in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model-00001.safetensors",
        ):
            (self.checkpoint / filename).write_text("{}")
        (self.checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model-00001.safetensors"}})
        )
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_root_selects_model_and_explicit_override_takes_priority(self):
        with patch.dict(os.environ, SMOKE_LOCAL_CHECKPOINT_ROOT=str(self.root)):
            self.assertEqual(
                resolve_local_checkpoint("DeepSeek-V4.1-Flash"), str(self.checkpoint)
            )
            self.assertEqual(
                resolve_local_checkpoint("other-model", str(self.checkpoint)),
                str(self.checkpoint),
            )

    def test_missing_root_has_no_nas_fallback(self):
        with self.assertRaisesRegex(ValueError, "NAS fallback is disabled"):
            resolve_local_checkpoint("DeepSeek-V4.1-Flash")

    def test_relative_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            resolve_local_checkpoint("model", "relative/model")

    def test_missing_shard_is_rejected(self):
        (self.checkpoint / "model-00001.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            resolve_local_checkpoint("model", str(self.checkpoint))

    def test_empty_shard_is_rejected(self):
        (self.checkpoint / "model-00001.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            resolve_local_checkpoint("model", str(self.checkpoint))

    def test_external_shard_symlink_is_rejected(self):
        external = self.root / "external.safetensors"
        external.write_bytes(b"weight")
        shard = self.checkpoint / "model-00001.safetensors"
        shard.unlink()
        shard.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "external"):
            resolve_local_checkpoint("model", str(self.checkpoint))

    def test_network_filesystem_is_rejected(self):
        for filesystem in ("nfs", "nfs4", "cifs", "smb2"):
            with self.subTest(filesystem=filesystem), patch(
                "rtp_llm.test.smoke.local_checkpoint.subprocess.check_output",
                return_value=filesystem + "\n",
            ):
                with self.assertRaisesRegex(ValueError, "local disk"):
                    resolve_local_checkpoint("model", str(self.checkpoint))

    def test_role_paths_change_without_changing_runtime_options(self):
        original = "/mnt/nas1/hf/model"
        checkpoint = "/local disk/model"
        arguments = {
            "prefill": f"--maxpander_enable 1 --redundant_expert 64 --sp_checkpoint_path {original}",
            "decode": f"--maxpander_enable 0 --redundant_expert 0 --sp_checkpoint_path={original}",
        }
        environment = {
            "prefill": [f"SP_CHECKPOINT_PATH={original}", "MAXPANDER_ENABLE=1"],
            "decode": [f"SP_CHECKPOINT_PATH={original}", "MAXPANDER_ENABLE=0"],
        }
        updated_args, updated_env = relocate_model_paths(
            arguments, environment, original, checkpoint
        )
        for role in arguments:
            expected_tokens = [
                token.replace(original, checkpoint)
                for token in shlex.split(arguments[role])
            ]
            self.assertEqual(shlex.split(updated_args[role]), expected_tokens)
            self.assertEqual(
                updated_env[role],
                [value.replace(original, checkpoint) for value in environment[role]],
            )
        self.assertIn(original, arguments["prefill"])
        self.assertEqual(environment["decode"][0], f"SP_CHECKPOINT_PATH={original}")

    def test_unrelated_paths_and_option_values_are_unchanged(self):
        arguments = "--sp_checkpoint_path /nas/model-other --flag 'a=b'"
        environment = ["PATH=/nas/model/bin", "OPTION=a=b"]
        updated_args, updated_env = relocate_model_paths(
            arguments, environment, "/nas/model", "/local/model"
        )
        self.assertEqual(shlex.split(updated_args), shlex.split(arguments))
        self.assertEqual(updated_env, environment)


if __name__ == "__main__":
    unittest.main()
