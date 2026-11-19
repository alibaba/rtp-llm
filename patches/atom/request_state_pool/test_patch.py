"""CPU source-fixture checks for the pinned additive dependency patch."""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "double_buffer_patch_apply", HERE / "apply.py"
)
APPLIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APPLIER)


class DoubleBufferPatchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="qwen4_double_buffer_patch_"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = json.loads((HERE / "manifest.json").read_text())
        for name, spec in self.manifest["components"].items():
            if env := os.getenv(f"RTP_DOUBLE_BUFFER_{name.upper()}_SOURCE"):
                source = Path(env)
            else:
                package = "atom" if name == "atom" else "amd-aiter"
                source = Path(importlib.metadata.distribution(package).locate_file(""))
            paths = spec["source_files_sha256"]
            for path in paths:
                original = source / (path[5:] if env and name == "atom" else path)
                if original.is_file():
                    target = self.root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original.read_bytes())

            # The image contains the postimage. Reconstruct the reviewed
            # preimage in scratch so both apply and rejection paths are tested.
            def hashes():
                return {path: APPLIER.digest(self.root / path) for path in paths}

            if not env and name == "atom" and hashes() not in (paths, spec["patched_files_sha256"]):
                # The image applies the shared native PD overlay last. Undo its
                # overlapping hunks in scratch before validating the pinned ABI.
                sidecar = HERE.parent / "main_0189b3506" / "native_pd_cache.patch"
                subprocess.run(
                    ["git", "apply", "--reverse", *(f"--include={path}" for path in paths), str(sidecar)],
                    cwd=self.root, check=True,
                )
            if hashes() == spec["patched_files_sha256"]:
                subprocess.run(
                    ["git", f"--work-tree={self.root}", "apply", "--reverse", str(HERE / spec["patch"])],
                    cwd=self.root, check=True,
                )
            self.assertEqual(hashes(), paths)

    def test_apply_verify_idempotence_and_compilable_sources(self):
        APPLIER.apply(self.root)
        APPLIER.apply(self.root)
        APPLIER.apply(self.root, verify_only=True)
        for spec in self.manifest["components"].values():
            for path in spec["patched_files_sha256"]:
                compile((self.root / path).read_text(), path, "exec")

    def test_unknown_source_rejected_before_any_component_is_modified(self):
        aiter = self.root / "aiter/ops/flydsl/kernels/gdr_decode.py"
        aiter.write_text(aiter.read_text() + "\n# unknown local edit\n")
        before = {
            path: (self.root / path).read_bytes()
            for path in self.manifest["components"]["atom"]["source_files_sha256"]
            if (self.root / path).exists()
        }
        with self.assertRaisesRegex(RuntimeError, "not the reviewed"):
            APPLIER.apply(self.root)
        for path, payload in before.items():
            self.assertEqual((self.root / path).read_bytes(), payload)
        self.assertFalse(
            (self.root / "atom/plugin/rtpllm/utils/linear_double_buffer.py").exists()
        )


if __name__ == "__main__":
    unittest.main()
