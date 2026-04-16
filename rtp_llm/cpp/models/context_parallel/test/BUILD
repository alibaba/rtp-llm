load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "th_context_parallel_py_wrapper_test_lib",
    srcs = [
        "zigzag_processor_py_wrapper_test.cc",
    ],
    hdrs = [
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/models:context_parallel",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_binary(
    name = "th_context_parallel_py_wrapper_test",
    deps = [":th_context_parallel_py_wrapper_test_lib"],
    linkshared = 1,
    visibility = ["//visibility:public"],
)


