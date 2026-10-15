"""B300-targeted diagnostic: dump the fmha_gen op's ninja graph and verdicts.

The minicpm SM103 smoke dies compiling the shared flashinfer cache's fmha_gen
op on a missing flashInferMetaInfo.h; the compile's -I anchor points at a
dead foreign-test runfiles sandbox, yet the runfiles-anchor eviction does not
fire. This probe runs inside the smoke-sm10x lane (SM103_CU13 pool) so its
stdout lands in the lane artifacts; it dumps the op's reachable ninja graph
verbatim plus the staleness verdicts from the eviction helper. TEMPORARY —
remove once the gap is root-caused.
"""
import glob
import unittest
from pathlib import Path

from rtp_llm.test.utils.jit_sys_path_setup import (
    _flashinfer_op_is_stale,
    _read_ninja_graph,
)


def _dump(op_dir: Path) -> None:
    print(f"[b300-probe] op: {op_dir}", flush=True)
    graph = _read_ninja_graph(op_dir)
    print(f"[b300-probe] graph bytes: {len(graph)}", flush=True)
    for line in graph.splitlines():
        if (
            "-I" in line
            or "-isystem" in line
            or ".runfiles" in line
            or "flashInferMetaInfo" in line
        ):
            print(f"[b300-probe] | {line[:240]}", flush=True)
    print(
        f"[b300-probe] stale verdict: {_flashinfer_op_is_stale(op_dir)}",
        flush=True,
    )


class B300CacheProbeTest(unittest.TestCase):
    def test_dump_fmha_gen_state(self):
        bases = glob.glob(
            "/tmp/rtp-llm/.jit_cache/v2/*/flashinfer/.cache/flashinfer/*/"
            "*/cached_ops/fmha_gen"
        )
        print(f"[b300-probe] fmha_gen op dirs: {bases}", flush=True)
        for op in sorted(Path(b) for b in bases):
            _dump(op)
        headers = glob.glob(
            "/tmp/rtp-llm/.jit_cache/v2/*/flashinfer/.cache/flashinfer/cubins/"
            "**/flashInferMetaInfo.h",
            recursive=True,
        )
        print(f"[b300-probe] metainfo headers found: {headers}", flush=True)
        print("[b300-probe] done", flush=True)


if __name__ == "__main__":
    unittest.main()
