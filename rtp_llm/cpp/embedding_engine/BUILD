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
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp:models",
        "//rtp_llm/cpp:schedulers",
        "//rtp_llm/cpp:engine_base",
        "//rtp_llm/cpp:utils",
        "//rtp_llm/cpp:normal_engine",
        "//rtp_llm/cpp:system_prompt",
        "//rtp_llm/cpp:system_prompt_constructor",
        ":embedding_handler_base",
        ":embedding_handler",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "embedding_handler_base",
    hdrs = [
        "handlers/HandlerBase.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    copts = copts(),
    visibility = ["//visibility:public"]
)

cc_library(
    name = "embedding_handler",
    hdrs = glob([
        "handlers/*.h",
    ], exclude = [
        "handlers/HandlerBase.h"
    ]),
    srcs = glob([
        "handlers/*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp:stream",
        ":embedding_handler_base",
        "//rtp_llm/cpp:dataclass",
        "//rtp_llm/cpp:models",
        "//:th_op_hdrs",
        "//rtp_llm/cpp:utils",
    ],
    copts = copts(),
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
        "@havenask//aios/network/arpc:arpc",
        "@havenask//aios/network/arpc/arpc/metric:kmonitor_anet_metric",
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
        "//rtp_llm/cpp:multimodal_processor",
    ],
    visibility = ["//visibility:public"]
)