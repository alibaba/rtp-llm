"""Short rank-local scratch paths, independent of persistent DeepGEMM caches."""

import hashlib
import os


def rank_nvcc_tmpdir(rank: int, namespace: str, override: str | None = None) -> str:
    # DeepGEMM starts MLOPart inside TMPDIR and appends
    # /deep_gemm_mps_<random>/control. Linux AF_UNIX paths must fit in 108 bytes.
    # Keep explicit short overrides usable, but never inherit a managed JIT
    # cache's arbitrarily deep scope path as the physical temporary directory.
    if override:
        candidate = os.path.join(override, namespace, f"rank_{int(rank)}")
        if len(os.fsencode(candidate)) + 40 < 108:
            return candidate
    scope = override or os.environ.get("DG_JIT_CACHE_DIR") or os.getcwd()
    digest = hashlib.sha256(os.fsencode(namespace + "\0" + scope)).hexdigest()[:16]
    return f"/tmp/rtp-dg-{os.getuid()}/{digest}-r{int(rank)}"
