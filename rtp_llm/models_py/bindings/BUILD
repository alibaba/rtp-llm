load("//:def.bzl", "copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "op_defs",
    hdrs = [
        "OpDefs.h",
        "OpDefsUtils.h",
        "ParamsBase.h",
    ],
    srcs = [
        "OpDefs.cc",
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/cache:cache_group_type",
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
    name = "no_block_copy_hdr",
    hdrs = ["NoBlockCopy.h"],
    deps = torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "no_block_copy_default",
    srcs = ["NoBlockCopyDefault.cc"],
    copts = copts(),
    deps = [
        ":no_block_copy_hdr",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
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
