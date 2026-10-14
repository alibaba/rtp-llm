"""Diagnostic probe: dump the shared flashinfer JIT cache's norm op state.

Runs on the H20 CI worker that hosts the poisoned norm op (the one whose
ninja build references a deleted .runfiles sandbox). Prints the op's ninja
graph verbatim and the staleness verdict from the eviction helper so the
detection gap can be root-caused from the test log. TEMPORARY — remove
after the diagnosis lands.
"""
import glob
import os
import unittest
from pathlib import Path

from rtp_llm.test.utils.jit_sys_path_setup import (
    _evict_dangling_flashinfer_ops,
    _flashinfer_op_is_stale,
)


class PoisonProbeTest(unittest.TestCase):
    def test_dump_norm_op_state(self):
        bases = glob.glob(
            "/tmp/rtp-llm/.jit_cache/v2/*/flashinfer/.cache/flashinfer/*/*/cached_ops"
        )
        print(f"[probe] cached_ops bases found: {bases}", flush=True)
        for base in bases:
            for op in sorted(Path(base).iterdir()):
                if not op.is_dir():
                    continue
                print(f"[probe] op: {op}", flush=True)
                for f in sorted(op.glob("*.ninja")):
                    print(f"[probe] --- {f.name} ---", flush=True)
                    print(f.read_text(errors="ignore"), flush=True)
                for f in sorted(op.rglob("*.ninja")):
                    if f.parent != op:
                        print(f"[probe] nested ninja: {f}", flush=True)
                        print(f.read_text(errors="ignore"), flush=True)
                stale = _flashinfer_op_is_stale(op)
                print(f"[probe] stale verdict: {op.name} -> {stale}", flush=True)
        # Also run the full eviction to observe its behavior on this worker.
        for scope in glob.glob("/tmp/rtp-llm/.jit_cache/v2/*/flashinfer"):
            print(f"[probe] evicting under {scope}", flush=True)
            _evict_dangling_flashinfer_ops(Path(scope))
        print("[probe] done", flush=True)


if __name__ == "__main__":
    unittest.main()
