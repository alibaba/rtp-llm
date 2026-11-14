load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "any_cuda_copts", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

any_cuda_deps = select({
    "@//:using_cuda": [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    "@//:using_rocm": [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr",
    ],
    "//conditions:default": [],
})

cc_test(
    name = "fuse_copy_kernel_test",
    srcs = ["fuse_copy_kernel_test.cc"],
    copts = any_cuda_copts(),
    deps = any_cuda_deps + [
        "//rtp_llm/models_py/bindings/common/kernels:fuse_copy_kernel",
        "@com_google_googletest//:gtest",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_test_wrapper(
    name = "tiled_temperature_test",
    srcs = ["TiledTemperatureTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/common/kernels:kernels_sampling",
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
