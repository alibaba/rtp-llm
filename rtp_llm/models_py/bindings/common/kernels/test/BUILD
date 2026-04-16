load("//:def.bzl", "cc_test_wrapper", "any_cuda_copts")

cc_test = cc_test_wrapper

any_cuda_deps = select({
    "@//:using_cuda": [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    "@//:using_rocm": [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:hip",
        "//rtp_llm/cpp/rocm:rocm_types_hdr",
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
    exec_properties = {"gpu": "H20"},
)
