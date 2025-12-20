load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "static_config",
    hdrs = ["StaticConfig.h"],
    srcs = ["StaticConfig.cc"],
    copts = copts(),
)

cc_library(
    name = "role_types",
    hdrs = ["RoleTypes.h"],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "config_modules",
    hdrs = ["ConfigModules.h"],
    srcs = ["ConfigModules.cc"],
    deps = [
        "@havenask//aios/autil:env_util",
        ":role_types",
        ":eplb_config",
    ],
    copts = copts(),
)

cc_library(
    name = "eplb_config",
    hdrs = ["EplbConfig.h"],
    srcs = ["EplbConfig.cc"],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "special_tokens",
    hdrs = ["SpecialTokens.h"],
    visibility = ["//visibility:public"],
    copts = copts(),
)

cc_library(
    name = "model_config",
    hdrs = ["ModelConfig.h"],
    srcs = ["ModelConfig.cc"],
    deps = [
        ":eplb_config",
        ":special_tokens",
        ":config_modules",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/core:buffer_torch"
    ],
    visibility = ["//visibility:public"],
    copts = copts(),
)