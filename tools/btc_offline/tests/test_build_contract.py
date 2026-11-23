"""Regression checks for backend.py auto. Run with unittest discovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/skills/btc-offline-replay/scripts"))
import backend


class BuildContract(unittest.TestCase):
    def setUp(self):
        self.marker = json.loads((ROOT / "btc-offline-backend.json").read_text())

    def test_missing_source_marker_cannot_apply_legacy_code_silently(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@invalid",
                            "commit", "-q", "--allow-empty", "-m", "empty source"], check=True)
            with self.assertRaisesRegex(ValueError, "source marker"):
                backend.auto_build(argparse.Namespace(repo=repo, work=Path(tmp) / "build"))

    def test_auto_rejects_model_graph_before_build(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "btc-offline-backend.json").write_text(json.dumps(self.marker))
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@invalid",
                            "commit", "-qm", "fixture marker"], check=True)
            bazel = root / "query-only"
            labels = self.marker["dependency_contract"]["require"] + ["//:rtp_compute_ops"]
            graph = "\n".join("cc_library rule " + label + " (12ab)" for label in labels)
            bazel.write_text("#!/bin/sh\n[ \"$1\" = cquery ] || exit 64\ncat <<'GRAPH'\n" + graph + "\nGRAPH\n")
            bazel.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "forbidden=.*//:rtp_compute_ops"):
                backend.auto_build(argparse.Namespace(repo=repo, work=root / "build",
                                                     bazel=str(bazel), option=[]))
            self.assertTrue((root / "build/dependencies.txt").is_file())
            self.assertFalse((root / "build/build-receipt.json").exists())

    def test_cannot_omit_native_lifecycle_check(self):
        self.marker["checks"].remove("//tools/btc_offline:metadata_replay_test")
        with self.assertRaisesRegex(ValueError, "metadata_replay_test"):
            backend.validate_fixed_recipe(self.marker)

    def test_graph_contract(self):
        rules = self.marker["dependency_contract"]
        def graph(labels):
            return "\n".join("cc_library rule " + label + " (12ab)" for label in labels)
        backend.check_graph(graph(rules["require"]), rules)
        for label in rules["require"]:
            with self.subTest(missing=label), self.assertRaisesRegex(ValueError, "missing"):
                backend.check_graph(graph(set(rules["require"]) - {label}), rules)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            backend.check_graph(graph(rules["require"] + ["//:rtp_compute_ops"]), rules)


if __name__ == "__main__":
    unittest.main()
