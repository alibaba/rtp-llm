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
        "//rtp_llm/cpp/config:eplb_config",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/models:context_parallel",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
    ] + select({
        "//:using_cuda": ["//rtp_llm/cpp/cuda_graph:cuda_graph_impl"],
        "@//:using_rocm": ["//rtp_llm/cpp/cuda_graph:cuda_graph_impl"],
        "//conditions:default": [],
    }),
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