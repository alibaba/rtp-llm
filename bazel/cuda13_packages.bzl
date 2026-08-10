# Distribution name to Bazel requirement wrapper target. CUDA 13 keeps these
# targets empty until compatible wheels are available.
# By design, this file does not own wheel URLs: source requirements own resolver
# inputs, generated locks own hashes, and whl_deps() owns packaged metadata.
# Keep overlapping entries synchronized until a generated manifest is adopted;
# this table is the single source only for CUDA 13 availability and smoke-test
# versions, not for the full dependency graph.
CUDA13_UNAVAILABLE_REQUIREMENTS = {
    "fast-hadamard-transform": "fast-hadamard-transform",
    "flash-attn": "flash_attn",
    "flash-attn-3": "flash-attn-3",
    "flashinfer-cubin": "flashinfer-cubin",
    "flashinfer-jit-cache": "flashinfer-jit-cache",
}

# Per-architecture qualification matrix (CUDA 13). x86 and ARM intentionally
# pin different dependency revisions; the authoritative values live in the
# CUDA13_EXPECTED_DEPENDENCY_VERSIONS table below (cutlass-dsl / tvm-ffi) and in
# whl_deps() / requirements_*cuda13*.txt (wheel URLs & SHAs). This block records
# WHY they diverge, not the values, so it does not become yet another copy:
#
#   nvidia-cutlass-dsl / apache-tvm-ffi : pinned per-arch until x86 and ARM are
#       qualified on a common release.
#   flash-mla / rtp-kernel wheels       : built from separately validated,
#       per-architecture platform revisions.
#
# Consequence: do NOT assume "x86 passed" implies "ARM passed" or vice versa.
# In particular the CuteDSL FP4 numerical regression runs only on the ARM SM100
# pool today (see rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/
# executors/test/BUILD), so x86 has no equivalent numerical coverage until an
# x86 SM100 pool + qualified wheels exist.
#
# TODO(<owner>): attach the tracking issue for converging x86/ARM onto one
# revision set and record the target convergence milestone here.

# Runtime dependency smoke tests consume these values. Keep them synchronized
# with the corresponding pins in the CUDA 13 source requirements files.
CUDA13_EXPECTED_DEPENDENCY_VERSIONS = {
    "arm": {
        "apache-tvm-ffi": "0.1.7",
        "nvidia-cutlass-dsl": "4.5.0",
    },
    "x86": {
        "apache-tvm-ffi": "0.1.10",
        "nvidia-cutlass-dsl": "4.4.2",
    },
}
