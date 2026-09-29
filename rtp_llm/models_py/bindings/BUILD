load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts", "rocm_copts")

cc_library(
    name = "op_defs",
    srcs = [
        "OpDefs.cc",
    ],
    hdrs = [
        "CacheStoreWriter.h",
        "OpDefs.h",
        "OpDefsUtils.h",
        "ParamsBase.h",
    ],
    visibility = ["//visibility:public"],
    deps = torch_deps() + [
        "//rtp_llm/cpp/cache:cache_group_type",
        "//rtp_llm/cpp/cache:cache_layout",
        "//rtp_llm/cpp/cache:kv_cache_specs",
        "//rtp_llm/cpp/model_utils",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    alwayslink = 1,
)

py_test(
    name = "py_model_inputs_compat_test",
    srcs = ["py_model_inputs_compat_test.py"],
    data = [
        "//rtp_llm:ops/librtp_compute_ops/__init__.pyi",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm:testlib",
    ],
)

cc_library(
    name = "register_ops_hdr",
    hdrs = [
        "RegisterOps.h",
    ],
    visibility = ["//visibility:public"],
    deps = torch_deps() + [
        ":op_defs",
    ],
)

cc_library(
    name = "no_block_copy_hdr",
    hdrs = ["NoBlockCopy.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = torch_deps(),
)

cc_library(
    name = "no_block_copy_default",
    srcs = ["NoBlockCopyDefault.cc"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":no_block_copy_hdr",
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    alwayslink = 1,
)

cc_library(
    name = "dummy_register",
    srcs = [
        "DummyRegister.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":register_ops_hdr",
    ],
    alwayslink = 1,
)
