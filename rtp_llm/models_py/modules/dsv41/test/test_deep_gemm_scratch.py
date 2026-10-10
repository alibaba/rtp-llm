"""Long managed JIT scopes must not overflow MLOPart's AF_UNIX address."""

import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.models_py.utils.deep_gemm_scratch import rank_nvcc_tmpdir


class DeepGemmScratchTest(unittest.TestCase):
    def test_long_managed_scope_can_bind_mps_socket(self):
        scope = "/managed/" + "scope/" * 80
        with patch.dict(os.environ, {"DG_JIT_CACHE_DIR": scope}):
            first = rank_nvcc_tmpdir(0, "mega")
            second = rank_nvcc_tmpdir(1, "mega")
            self.assertNotEqual(first, second)
            self.assertEqual(os.environ["DG_JIT_CACHE_DIR"], scope)
            Path(first).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="deep_gemm_mps_", dir=first
            ) as pipe:
                with socket.socket(socket.AF_UNIX) as server:
                    server.bind(str(Path(pipe) / "control"))
            Path(first).rmdir()
            with patch.dict(os.environ, {"DG_JIT_CACHE_DIR": scope + "other"}):
                self.assertNotEqual(first, rank_nvcc_tmpdir(0, "mega"))

    def test_explicit_long_override_is_shortened(self):
        path = rank_nvcc_tmpdir(3, "mega", "/long/" + "nested/" * 80)
        self.assertLess(len(os.fsencode(path)) + 40, 108)
        self.assertEqual(
            rank_nvcc_tmpdir(3, "mega", "/tmp/test"), "/tmp/test/mega/rank_3"
        )


if __name__ == "__main__":
    unittest.main()
