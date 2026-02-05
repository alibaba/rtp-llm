load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "normal_engine",
    hdrs = glob([
        "*.h",
        "speculative/*.h",
    ]),
    srcs = glob([
        "*.cc",
        "speculative/*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base:worker_status_info",
        "//rtp_llm/cpp/engine_base:profiler",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt_constructor",
        "//rtp_llm/cpp/models:eplb",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "normal_engine_arpc_server_header",
    hdrs = [
        "arpc/ArpcServiceCreator.h",
    ],
    deps = [
        "//rtp_llm/cpp/engine_base:engine_base"
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "normal_engine_arpc_server_impl",
    srcs = glob([
        "arpc/ArpcServiceCreator.cc"
    ]),
    deps = [
        "@havenask//aios/network/arpc:arpc",
        "//rtp_llm/cpp/engine_base:engine_base",
        ":normal_engine_arpc_server_header",
    ],
    visibility = ["//visibility:public"]
)