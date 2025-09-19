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
    name = "config_modules",
    hdrs = ["ConfigModules.h"],
    srcs = ["ConfigModules.cc"],
    deps = [
        "@havenask//aios/autil:env_util",
    ],
    copts = copts(),
)

cc_library(
    name = "gpt_init_params",
    srcs = [
        "GptInitParameter.cc",
    ],
    hdrs = [
        "GptInitParameter.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/core:types",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/models:eplb_config",
    ] + torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)

