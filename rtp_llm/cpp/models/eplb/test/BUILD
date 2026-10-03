load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "th_eplb_py_wrapper_test_lib",
    srcs = [
        "eplb_py_wrapper_test.cc",
    ],
    hdrs = [
    ],
    copts = cuda_copts(),
    deps = torch_deps() + [
        "//rtp_llm/cpp/models:eplb",
        "//rtp_llm/cpp/pybind:th_compute_lib",
        "//rtp_llm/cpp/pybind:th_utils",
    ],
    alwayslink = True,
)

cc_binary(
    name = "th_eplb_py_wrapper_test",
    linkshared = 1,
    visibility = ["//visibility:public"],
    deps = [":th_eplb_py_wrapper_test_lib"],
)

py_test(
    name = "eplb_py_wrapper_test",
    srcs = [
        "eplb_py_wrapper_test.py",
    ],
    data = [
        ":th_eplb_py_wrapper_test",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm:torch",
    ],
)
