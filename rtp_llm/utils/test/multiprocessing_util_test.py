import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MultiprocessingTempDirTest(unittest.TestCase):
    def test_tensor_round_trip_with_long_temp_dir(self):
        script = """
import json
import multiprocessing
import multiprocessing.util
import tempfile
import torch
from rtp_llm.utils.multiprocessing_util import ensure_short_ipc_temp_dir

original = tempfile.gettempdir()
directory = ensure_short_ipc_temp_dir()
with multiprocessing.get_context('spawn').Pool(
    1, initializer=ensure_short_ipc_temp_dir
) as pool:
    value = torch.arange(5)
    result = pool.apply_async(torch.neg, (value,)).get(timeout=30)
    assert torch.equal(result, -value), result
assert tempfile.gettempdir() == original
assert ensure_short_ipc_temp_dir() == directory
print(json.dumps({'directory': directory, 'original': original}))
"""
        with tempfile.TemporaryDirectory(dir="/tmp") as root:
            for name in ("short", "long_" + "x" * 130):
                with self.subTest(name=name):
                    temporary = Path(root) / name
                    temporary.mkdir()
                    result = subprocess.run(
                        [sys.executable, "-c", script],
                        env={**os.environ, "TMPDIR": str(temporary)},
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    info = json.loads(result.stdout.splitlines()[-1])
                    self.assertEqual(info["original"], str(temporary))
                    self.assertFalse(Path(info["directory"]).exists())


if __name__ == "__main__":
    unittest.main()
