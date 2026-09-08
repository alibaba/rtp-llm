# Description:
#   CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance matrix-matrix
#   multiplication (GEMM) and related computations at all levels and scales within CUDA.
licenses([
    "notice",  # Portions BSD
])

CUTLASS_FILES = [
    "include/**",
]

# Files known to be under MPL2 license.
CUTLASS_HEADER_FILES = glob(
    CUTLASS_FILES,
)

cc_library(
    name = "cutlass_utils",
    hdrs = glob([
        "tools/util/include/**/*.h",
        "tools/util/include/**/*.hpp",
        "tools/util/include/**/*.cuh"
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    strip_include_prefix = "tools/util/include/",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = CUTLASS_HEADER_FILES,
    deps = [":cutlass_utils"],
    strip_include_prefix = "include",
    includes = ["."],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cutlass_origin",
    srcs = CUTLASS_HEADER_FILES,
    visibility = ["//visibility:public"]
)