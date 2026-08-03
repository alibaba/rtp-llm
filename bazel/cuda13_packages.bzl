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
