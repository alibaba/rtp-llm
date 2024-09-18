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
        "//maga_transformer/cpp:dataclass",
        "//maga_transformer/cpp:models",
        "//maga_transformer/cpp:schedulers",
        "//maga_transformer/cpp:engine_base",
        "//maga_transformer/cpp:common",
        "//maga_transformer/cpp:normal_engine",
        "//maga_transformer/cpp:system_prompt",
        "//maga_transformer/cpp:system_prompt_constructor",
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
        "//maga_transformer/cpp:stream",
        ":embedding_handler_base",
        "//maga_transformer/cpp:dataclass",
        "//maga_transformer/cpp:models",
        "//:th_op_hdrs",
        "//src/fastertransformer/utils:utils",
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
        ":embedding_engine_arpc_server_header"
    ],
    visibility = ["//visibility:public"]
)