load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

cc_test(
    name = "torch_profiler_test",
    srcs = ["TorchProfilerTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base:profiler",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "FT_SERVER_TEST": "1",
    },
)

cc_test(
    name = "cupti_trap_coredump_repro_test",
    srcs = ["CuptiTrapCoredumpReproTest.cu"],
    copts = cuda_copts(),
    deps = [
        "//rtp_llm/cpp/engine_base:profiler",
        "@com_google_googletest//:gtest",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    env = {
        "FT_SERVER_TEST": "1",
    },
    tags = ["manual"],
    timeout = "moderate",
)
