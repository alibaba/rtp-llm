import os
import subprocess
import sys
from unittest import TestCase, main

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class TestTopLevelTorchPatch(TestCase):
    def test_import_rtp_llm_installs_patch_without_loading_ops(self):
        code = """
import sys
import torch
import torch.distributed as dist
original_broadcast = dist.broadcast
import rtp_llm

patch = sys.modules.get("rtp_llm.utils.torch_patch")
assert patch is not None
assert torch.concat is patch.custom_concat
if hasattr(torch, "float8_e8m0fnu"):
    assert dist.broadcast is patch._ue8m0_broadcast
else:
    assert dist.broadcast is original_broadcast
assert "rtp_llm.ops" not in sys.modules
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (_REPO, env.get("PYTHONPATH", "")) if path
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_import_without_ue8m0_keeps_cpu_broadcast(self):
        code = """
import sys
import torch
import torch.distributed as dist
if hasattr(torch, "float8_e8m0fnu"):
    del torch.float8_e8m0fnu
original_broadcast = dist.broadcast
import rtp_llm
patch = sys.modules["rtp_llm.utils.torch_patch"]
assert torch.concat is patch.custom_concat
assert dist.broadcast is original_broadcast
assert "rtp_llm.ops" not in sys.modules
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (_REPO, env.get("PYTHONPATH", "")) if path
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    main()
