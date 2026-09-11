load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
    "//conditions:default": [],
})

cc_test(
    name = "zigzag_token_layout_test",
    srcs = [
        "ZigzagTokenLayoutTest.cc",
    ],
    deps = [
        "//rtp_llm/cpp/models:context_parallel_layout",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

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
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/core:op_data",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_binary(
    name = "th_context_parallel_py_wrapper_test",
    deps = [":th_context_parallel_py_wrapper_test_lib"] + cuda13_torch_link_deps,
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_test(
    name = "context_parallel_py_wrapper_test",
    srcs = [
        "context_parallel_py_wrapper_test.py"
    ],
    data = [
        ":th_context_parallel_py_wrapper_test"
    ],
    deps = [
        "//rtp_llm:torch"
    ],
    exec_properties = {'gpu':'H20'},
)
