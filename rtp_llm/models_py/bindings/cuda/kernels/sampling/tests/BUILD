load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test_wrapper(
    name = "split_softmax_test",
    srcs = ["SplitSoftmaxTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling:softmax_host",
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling:split_softmax",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)

renorm_test_deps = [
    "//rtp_llm/models_py/bindings/cuda/kernels/sampling:sampling",
    "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_test_wrapper(
    name = "renorm_test",
    srcs = ["RenormTest.cc"],
    copts = copts(),
    deps = renorm_test_deps,
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)

cc_test_wrapper(
    name = "renorm_sm103_test",
    srcs = ["RenormTest.cc"],
    copts = copts(),
    deps = renorm_test_deps,
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    tags = ["SM103_CU13"],
)

cc_library(
    name = "air_top_p_bench_helper",
    srcs = ["AirTopPBenchHelper.cu"],
    hdrs = ["AirTopPBenchHelper.h"],
    copts = cuda_copts(),
    deps = [
        "//rtp_llm/models_py/bindings/cuda/kernels/sampling:sampling",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sampling_benchmark_deps = [
    ":air_top_p_bench_helper",
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
    "//rtp_llm/models_py/bindings/cuda/kernels/sampling:sampling",
    "//rtp_llm/models_py/bindings/cuda/kernels/sampling:softmax_host",
    "//rtp_llm/models_py/bindings/cuda/kernels/sampling:temperature_host",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_test_wrapper(
    name = "sampling_benchmark_test",
    srcs = ["SamplingBenchmarkTest.cc"],
    copts = copts(),
    deps = sampling_benchmark_deps,
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13", "manual", "perf"],
)

cc_test_wrapper(
    name = "sampling_benchmark_sm103_test",
    srcs = ["SamplingBenchmarkTest.cc"],
    copts = copts(),
    deps = sampling_benchmark_deps,
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    tags = ["SM103_CU13", "manual", "perf"],
)
