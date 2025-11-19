load("//:def.bzl", "copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "op_defs",
    hdrs = [
        "OpDefs.h",
        "OpDefsUtils.h",
        "ParamsBase.h",
        "MlaParamsBase.h",
    ],
    srcs = [
        "OpDefs.cc",
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "register_ops_hdr",
    hdrs = [
        "RegisterOps.h",
    ],
    deps = torch_deps() + [
        ":op_defs",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dummy_register",
    srcs = [
        "DummyRegister.cc",
    ],
    deps = [
        ":register_ops_hdr",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
