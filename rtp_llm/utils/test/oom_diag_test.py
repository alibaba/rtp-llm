"""Unit tests for rtp_llm.utils.oom_diag.

Coverage:
  - default (env unset): install_oom_dump() is a no-op -- no observer,
    no recording, no per-block frames captured
  - RTP_OOM_RECORD=1: installs observer, enables full history recording
    with python stacks; per-block frames are populated for active blocks
  - observer writes marker + snapshot files; observer is one-shot per
    process; install is idempotent
  - filenames embed rank + pid (multi-process safety)
"""

import os
import pickle
import shutil
import tempfile
import unittest

import torch

from rtp_llm.utils import oom_diag


def _reset_module_state() -> None:
    oom_diag._installed = False
    oom_diag._oom_fired = False
    torch.cuda.memory._record_memory_history(enabled=None)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class OomDiagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="oom_diag_test_")
        self._saved_record_env = os.environ.get(oom_diag._RECORD_ENV)
        self._saved_out_dir = oom_diag._OUT_DIR
        oom_diag._OUT_DIR = self.tmp
        os.environ.pop(oom_diag._RECORD_ENV, None)
        _reset_module_state()

    def tearDown(self) -> None:
        _reset_module_state()
        oom_diag._OUT_DIR = self._saved_out_dir
        if self._saved_record_env is None:
            os.environ.pop(oom_diag._RECORD_ENV, None)
        else:
            os.environ[oom_diag._RECORD_ENV] = self._saved_record_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- helpers ----

    def _active_block_frames(self) -> list:
        """Return frames lists for every currently-active block across all
        segments. Empty list per block when recording is off."""
        snap = torch.cuda.memory._snapshot()
        out = []
        for seg in snap.get("segments", []):
            for blk in seg.get("blocks", []):
                if blk.get("state") == "active_allocated":
                    out.append(blk.get("frames", []))
        return out

    def _list_outputs(self):
        files = os.listdir(self.tmp)
        markers = sorted(f for f in files if f.startswith("OOM_MARKER_"))
        snaps = sorted(
            f for f in files if f.startswith("snap_") and f.endswith(".pickle")
        )
        return markers, snaps

    # ---- default behavior ----

    def test_default_install_is_noop(self) -> None:
        """No env -> install does nothing: no recording, no observer flag."""
        oom_diag.install_oom_dump()
        self.assertFalse(oom_diag._enabled())
        self.assertFalse(oom_diag._installed)

        x = torch.zeros(8192, dtype=torch.float32, device="cuda")
        try:
            for frames in self._active_block_frames():
                self.assertEqual(frames, [])
        finally:
            del x
            torch.cuda.empty_cache()

    # ---- enabled behavior ----

    def test_enabled_records_python_stacks(self) -> None:
        """RTP_OOM_RECORD=1 -> active blocks carry Python stacks."""
        os.environ[oom_diag._RECORD_ENV] = "1"
        oom_diag.install_oom_dump()
        self.assertTrue(oom_diag._enabled())
        self.assertTrue(oom_diag._installed)

        x = torch.zeros(8192, dtype=torch.float32, device="cuda")
        try:
            frames_per_block = self._active_block_frames()
            self.assertTrue(
                any(len(f) > 0 for f in frames_per_block),
                "expected at least one active block to have python frames",
            )
            joined = " ".join(
                str(frame) for frames in frames_per_block for frame in frames
            )
            self.assertIn("oom_diag_test", joined)
        finally:
            del x
            torch.cuda.empty_cache()

    def test_enabled_records_alloc_history(self) -> None:
        """RTP_OOM_RECORD=1 enables full-mode recording -> snapshot's
        device_traces contains alloc / free events (not just OOM)."""
        os.environ[oom_diag._RECORD_ENV] = "1"
        oom_diag.install_oom_dump()

        x = torch.zeros(8192, dtype=torch.float32, device="cuda")
        try:
            snap = torch.cuda.memory._snapshot()
            traces = snap.get("device_traces", [])
            alloc_events = sum(
                1 for tr in traces for e in tr if e.get("action") == "alloc"
            )
            self.assertGreater(
                alloc_events,
                0,
                "expected at least one alloc event in device_traces under full-mode recording",
            )
        finally:
            del x
            torch.cuda.empty_cache()

    def test_install_is_idempotent(self) -> None:
        os.environ[oom_diag._RECORD_ENV] = "1"
        oom_diag.install_oom_dump()
        oom_diag.install_oom_dump()
        self.assertTrue(oom_diag._installed)

    # ---- observer behavior ----

    def test_observer_writes_marker_and_snapshot(self) -> None:
        os.environ[oom_diag._RECORD_ENV] = "1"
        oom_diag.install_oom_dump()
        oom_diag._oom_observer(
            device=0, alloc_size=4096, device_total=80 * 1024**3, device_free=1024
        )

        markers, snaps = self._list_outputs()
        self.assertEqual(len(markers), 1, f"markers={markers}")
        self.assertEqual(len(snaps), 1, f"snaps={snaps}")

        with open(os.path.join(self.tmp, markers[0])) as f:
            marker_text = f.read()
        self.assertIn("failed_alloc=4096", marker_text)
        self.assertIn("device=0", marker_text)
        self.assertIn(f"pid={os.getpid()}", marker_text)
        self.assertIn("stack=", marker_text)

        with open(os.path.join(self.tmp, snaps[0]), "rb") as f:
            snap = pickle.load(f)
        self.assertIn("segments", snap)

    def test_observer_is_one_shot_per_process(self) -> None:
        """Second invocation must not write more files."""
        os.environ[oom_diag._RECORD_ENV] = "1"
        oom_diag.install_oom_dump()
        oom_diag._oom_observer(0, 4096, 80 * 1024**3, 1024)
        oom_diag._oom_observer(0, 8192, 80 * 1024**3, 2048)
        markers, snaps = self._list_outputs()
        self.assertEqual(len(markers), 1)
        self.assertEqual(len(snaps), 1)

    # ---- multi-process file naming ----

    def test_suffix_includes_device_and_pid(self) -> None:
        suffix = oom_diag._suffix("test", 7)
        self.assertIn("_d7_", suffix)
        self.assertIn(f"_pid{os.getpid()}_", suffix)
        self.assertTrue(suffix.startswith("test_"))


if __name__ == "__main__":
    unittest.main()
