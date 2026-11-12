load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test_wrapper(
    name = "split_softmax_test",
    srcs = ["SplitSoftmaxTest.cc"],
    copts = copts(),
    deps = [
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
