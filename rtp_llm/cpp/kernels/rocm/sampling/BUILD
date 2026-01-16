load("//:def.bzl", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "sampling",
    srcs = ["api.cc"],
    hdrs = [
        "kernel.cuh",
        "sampling.h",
        "utils.h",
    ],
    deps = [
        "//rtp_llm/cpp/kernels:rocm_utils",
        "@local_config_rocm//rocm:rocm_headers",
    ] + torch_deps(),
    copts = rocm_copts() + [
        "-DUSE_ROCM=1",
        "-DFLASHINFER_CUB_SUBTRACTLEFT_DEFINED",  # hipCUB 支持 SubtractLeft 而不是 FlagHeads
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "bind.so",
    srcs = [
        "bind.cc",
    ],
    deps = [
        ":sampling",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
    linkshared = True,
    linkstatic = False,
)

# py_test(
#     name = "test",
#     srcs = ["test.py"],
#     data = [
#         ":bind.so",
#     ],
#     imports = ["."],
#     python_version = "PY3",
#     tags = ["rocm"],
#     exec_properties = {'gpu':'MI308X'}
# )