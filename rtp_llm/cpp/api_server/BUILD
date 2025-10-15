load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "embedding_arpc_deps")

embedding_arpc_deps()

# Tokenizer - text tokenization component
cc_library(
    name = "tokenizer",
    hdrs = [
        "tokenizer/Tokenizer.h",
    ],
    srcs = [
        "tokenizer/Tokenizer.cc",
    ],
    deps = [
        "//rtp_llm/cpp/pybind:py_utils",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

# OpenAI API implementation
cc_library(
    name = "openai",
    hdrs = [
        "openai/ApiDataType.h",
        "openai/ChatRender.h",
        "openai/OpenaiEndpoint.h",
    ],
    srcs = [
        "openai/ApiDataType.cc",
        "openai/ChatRender.cc",
        "openai/OpenaiEndpoint.cc",
    ],
    deps = [
        ":tokenizer",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        ":embedding_arpc_deps",
        "//rtp_llm/cpp/multimodal_processor:multimodal_processor",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "http_api_server",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        ":openai",
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/engine_base:weights_converter",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/normal_engine:normal_engine",
        "//rtp_llm/cpp/embedding_engine:embedding_engine",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/multimodal_processor:multimodal_processor",
        ":http_server",
        ":http_client",
        ":embedding_arpc_deps",
        "//rtp_llm/cpp/api_server/common:common_service",
    ],
    visibility = ["//visibility:public"],
)

# HTTP Server components
cc_library(
    name = "http_server",
    srcs = glob(["http_server/http_server/*.cpp"]),
    hdrs = glob(["http_server/http_server/*.h"]),
    include_prefix = "http_server",
    strip_include_prefix = "http_server/http_server",
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "@havenask//aios/network/arpc:arpc",
    ],
)

cc_library(
    name = "http_client",
    srcs = glob(["http_server/http_client/*.cpp"]),
    hdrs = glob(["http_server/http_client/*.h"]),
    include_prefix = "http_client",
    strip_include_prefix = "http_server/http_client",
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "@havenask//aios/network/arpc:arpc",
    ],
)

cc_test(
    name = "http_client_gtest",
    srcs = [
        "http_server/test/main.cpp",
        "http_server/test/HttpClientTest.cpp",
        "http_server/test/ConnectionPoolTest.cpp"
    ],
    deps = [
        ":http_client",
        ":http_server",
        "@havenask//aios/autil:net",
        "//rtp_llm/cpp/utils:core_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@havenask//aios/network/arpc:arpc",
    ],
    copts = ["-fno-access-control",],
)
