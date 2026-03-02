load("//:def.bzl", "copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "dcu_bindings",
    srcs = glob([
        "*.cc",
    ], exclude = [
        "RegisterDcuOps.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    copts = rocm_copts(),
    deps = torch_deps() + [
        "//rtp_llm/cpp/core:types_hdr",
        "//rtp_llm/models_py/bindings/common/kernels:kernels_kv_cache",
	"//rtp_llm/models_py/bindings/dcu/kernels:rocm_cu",
	"//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/models_py/bindings:op_defs",
        "//rtp_llm/models_py/bindings/common:common",
    ],
)

cc_library(
    name = "dcu_bindings_base",
    hdrs = [
        "RegisterBaseBindings.hpp",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":dcu_bindings",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "dcu_bindings_register",
    srcs = [
        "RegisterDcuOps.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        ":dcu_bindings_base",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

