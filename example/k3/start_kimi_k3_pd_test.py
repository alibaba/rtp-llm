import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


class StartKimiK3PdTest(unittest.TestCase):
    def run_dry_run(self, role: str, **overrides: str) -> str:
        script = pathlib.Path(__file__).with_name("start_kimi_k3_pd.sh")
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        bash_major = int(
            subprocess.check_output(
                [bash, "-c", "printf %s \"${BASH_VERSINFO[0]}\""], text=True
            )
        )
        if bash_major < 4:
            self.skipTest("start_kimi_k3_pd.sh requires Bash 4+")
        with tempfile.TemporaryDirectory() as root:
            root_path = pathlib.Path(root)
            model_path = root_path / "model"
            model_path.mkdir()
            (model_path / "config.json").touch()
            (model_path / "model.safetensors.index.json").touch()

            env = os.environ.copy()
            env.pop("THINK_START_TAG", None)
            env.pop("THINK_END_TAG", None)
            env.update(
                {
                    "CHECKPOINT_PATH": str(model_path),
                    "TOKENIZER_PATH": str(model_path),
                    "PREFILL_ENDPOINT": "127.0.0.1:27188",
                    "DECODE_ENDPOINT": "127.0.0.1:29188",
                    "RUN_ROOT": str(root_path / "run"),
                    "RTP_LLM_TMPDIR": str(root_path / "tmp"),
                    "RTP_LLM_SERVER_BINARY": "/bin/true",
                    "RTP_LLM_SKIP_BUILD": "1",
                    "RTP_LLM_DRY_RUN": "1",
                }
            )
            env.update(overrides)
            result = subprocess.run(
                [bash, str(script), role],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout

    def test_defaults_to_complete_k3_think_boundary_for_both_roles(self) -> None:
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                output = self.run_dry_run(role)
                self.assertIn("think start:     <|open|>think<|sep|>", output)
                self.assertIn(
                    "think end:       <|close|>think<|sep|><|open|>response<|sep|>",
                    output,
                )

    def test_preserves_explicit_think_boundary_override(self) -> None:
        output = self.run_dry_run(
            "prefill",
            THINK_START_TAG="custom-start",
            THINK_END_TAG="custom-end",
        )
        self.assertIn("think start:     custom-start", output)
        self.assertIn("think end:       custom-end", output)


if __name__ == "__main__":
    unittest.main()
