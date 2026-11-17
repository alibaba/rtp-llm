"""CPU source-fixture checks for the pinned additive dependency patch."""

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOM_SOURCE = Path(
    os.getenv(
        "RTP_DOUBLE_BUFFER_ATOM_SOURCE",
        str(HERE / "baseline_atom"),
    )
)
AITER_SOURCE = Path(
    os.getenv("RTP_DOUBLE_BUFFER_AITER_SOURCE", str(HERE / "baseline_aiter"))
)
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
            for path, sha in spec["source_files_sha256"].items():
                if sha == hashlib.sha256(b"").hexdigest():
                    continue
                source = (
                    ATOM_SOURCE / path[5:] if name == "atom" else AITER_SOURCE / path
                )
                self.assertTrue(
                    source.is_file(),
                    f"Required reviewed source fixture missing: {source}",
                )
                payload = source.read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), sha)
                target = self.root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)

    def test_apply_verify_idempotence_and_compilable_sources(self):
        APPLIER.apply(self.root)
        APPLIER.apply(self.root)
        APPLIER.apply(self.root, verify_only=True)
        for spec in self.manifest["components"].values():
            for path in spec["patched_files_sha256"]:
                compile((self.root / path).read_text(), path, "exec")
        self.assertEqual(
            (
                self.root / "atom/plugin/rtpllm/utils/linear_double_buffer.py"
            ).read_bytes(),
            (HERE / "runtime_adapter.py").read_bytes(),
        )

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
