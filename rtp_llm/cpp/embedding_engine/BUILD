load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "requirement", "whl_deps")

cc_library(
    name = "embedding_engine",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/engine_base:executor",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base:worker_status_info",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/cache:cache",
    ],
    visibility = ["//visibility:public"]
)


cc_library(
    name = "embedding_engine_arpc_server_header",
    hdrs = glob([
        "arpc/ArpcServerWrapper.h",
        "arpc/ArpcServiceCreator.h"
    ]),
    srcs = glob([
        "arpc/ArpcServerWrapper.cc"
    ]),
    deps = [
        ":embedding_engine"
    ],
    visibility = ["//visibility:public"]

)

cc_library(
    name = "embedding_engine_arpc_server_impl",
    srcs = glob([
        "arpc/ArpcServiceCreator.cc"
    ]),
    deps = [
        "@havenask//aios/network/arpc:arpc",
        ":embedding_engine",
        ":embedding_engine_arpc_server_header",
        "//rtp_llm/cpp/multimodal_processor:multimodal_processor",
    ],
    visibility = ["//visibility:public"]
)