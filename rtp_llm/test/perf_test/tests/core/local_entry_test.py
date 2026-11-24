"""Local Bazel entry preserves inputs while retaining relative outputs."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test import local_main
from rtp_llm.test.perf_test.tests.support.fixtures import fixture_config


class LocalEntryTest(unittest.TestCase):
    def test_dry_run_resolves_config_before_chdir_and_retains_outputs(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            (root / "config.json").write_text(json.dumps(fixture_config()))
            os.chdir(root)
            try:
                for arguments in (
                    ["--config=config.json"],
                    ["--config", "config.json"],
                ):
                    with self.subTest(arguments=arguments):
                        stdout = io.StringIO()
                        with patch.dict(
                            os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": str(outputs)}
                        ), redirect_stdout(stdout):
                            code = local_main.main(
                                [
                                    *arguments,
                                    "--mode=prefill",
                                    "--action=run",
                                    "--dry-run",
                                ]
                            )
                        self.assertEqual(0, code)
                        plan = json.loads(stdout.getvalue())
                        self.assertTrue(
                            Path(plan["result_directory"]).is_relative_to(outputs)
                        )
                        self.assertEqual(root, Path.cwd())
                        self.assertFalse((outputs / "perf_test_results").exists())
            finally:
                os.chdir(original)

    def test_absolute_output_root_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    fixture_config(output={"result_root": str(root / "results")})
                )
            )
            stdout = io.StringIO()
            with patch.dict(
                os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": directory}
            ), redirect_stdout(stdout):
                self.assertEqual(
                    0, local_main.main(["--config=" + str(config), "--dry-run"])
                )
            self.assertTrue(
                Path(json.loads(stdout.getvalue())["result_directory"]).is_relative_to(
                    root / "results"
                )
            )

    def test_run_directory_and_context_are_resolved_and_exit_code_forwarded(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            for args in (
                ["--run-dir=run", "--run-context=context.json", "--action=report"],
                [
                    "--run-dir",
                    "run",
                    "--run-context",
                    "context.json",
                    "--action=report",
                ],
            ):
                with self.subTest(args=args), patch.dict(
                    os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": directory}
                ):

                    def execute(argv):
                        from rtp_llm.test.perf_test.cli import parse_cli

                        request = parse_cli(argv)
                        self.assertEqual(str(original / "run"), request.run_dir)
                        self.assertEqual(
                            str(original / "context.json"), request.run_context
                        )
                        self.assertEqual(Path(directory), Path.cwd())
                        return 4

                    with patch.object(local_main, "perf_main", side_effect=execute):
                        self.assertEqual(4, local_main.main(args))
                self.assertEqual(original, Path.cwd())

    def test_missing_config_value_keeps_shared_cli_diagnostic(self):
        errors = io.StringIO()
        from contextlib import redirect_stderr

        with redirect_stderr(errors):
            self.assertEqual(2, local_main.main(["--config", "--dry-run"]))
        self.assertIn("expected one argument", errors.getvalue())

    def test_exception_restores_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": directory}
        ):
            with patch.object(
                local_main, "perf_main", side_effect=RuntimeError("failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    local_main.main(["--help"])
        self.assertEqual(original, Path.cwd())


if __name__ == "__main__":
    unittest.main()
