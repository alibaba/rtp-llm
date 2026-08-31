load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

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
    name = "multimodal_processor_test",
    srcs = glob([
        "MultimodalProcessorTest.cc",
        "FakeMultimodalProcessor.h",
    ]),
    copts = test_copts,
    deps = torch_deps() + cuda13_torch_link_deps + [
        "//rtp_llm/cpp/testing:device_test_utils",

        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//rtp_llm/cpp/multimodal_processor:multimodal_processor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)
