load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "th_eplb_py_wrapper_test_lib",
    srcs = [
        "eplb_py_wrapper_test.cc",
    ],
    hdrs = [
    ],
    deps = torch_deps() + [
        "//maga_transformer/cpp:normal_engine",
        "//:th_utils",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_binary(
    name = "th_eplb_py_wrapper_test",
    deps = [":th_eplb_py_wrapper_test_lib"],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_test(
    name = "eplb_py_wrapper_test",
    srcs = [
        "eplb_py_wrapper_test.py"
    ],
    data = [
        ":th_eplb_py_wrapper_test"
    ],
    deps = [
        "//maga_transformer:torch"
    ],
    exec_properties = {'gpu':'A10'},
)