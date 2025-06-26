load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "register_ops_hdr",
    hdrs = [
        "RegisterOps.h",
    ],
    deps = torch_deps(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dummy_register",
    hdrs = [
        "DummyRegister.cc",
    ],
    deps = [
        ":register_ops_hdr",
    ],
    visibility = ["//visibility:public"],
)
