load("//:def.bzl", "cuda_copts")

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lcuda",
    "-lnccl",
    "-lnvToolsExt",
]

cc_binary(
    name = "gemm_perf",
    srcs = [
        # "fpa_intb_perf.cu",
        "weight_only_batched_gemv.cu"
    ],
    deps = [
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/utils:utils_for_3rdparty",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_sample_helpers",
    hdrs = [
        "helper_cuda.h",
        "helper_string.h",
    ],
    strip_include_prefix = "",
)

cc_test(
    name = "gemm_runner_test",
    srcs = [
        "gemm_runner_test.cc",
    ],
    deps = [
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/layers:layers",
        "@com_google_googletest//:gtest"
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
    linkopts = test_linkopts,
    linkstatic = 1,
)
