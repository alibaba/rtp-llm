load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

test_copts = [
    "-fno-access-control",
] + copts()

test_linkopts = [
    "-lpython3.10",
    "-ltorch",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch_python",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
    "-lnccl",
    "-lnvToolsExt",
] + select({
    "@//:using_cuda": [
        "-ltorch_cuda",
        "-lc10_cuda",
    ],
    "@//:using_rocm": [
        "-ltorch_hip",
        "-lc10_hip",
    ],
    "//conditions:default": [
    ],
})

cc_test(
    name = "multimodal_processor_test",
    srcs = glob([
        "MultimodalProcessorTest.cc",
    ]),
    copts = test_copts,
    linkopts = test_linkopts,
    deps = torch_deps() + [
        "//rtp_llm/cpp/devices/testing:device_test_utils",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        "//:gpt_init_params",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//rtp_llm/cpp:multimodal_processor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'A10'},
)
