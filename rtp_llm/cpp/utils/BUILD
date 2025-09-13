load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//:__subpackages__"])

cc_library(
    name = "utils",
    hdrs = glob([
        "*.h",
    ], exclude=[
        "RpcErrorCode.h",
        "PyUtils.h",
    ]),
    srcs = glob([
        "*.cc",
    ], exclude=[
        "PyUtils.cc",
    ]),
    deps = [
        "@havenask//aios/alog:alog",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:net",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
    ],
    implementation_deps = [
        "//rtp_llm/cpp/config:static_config",
    ],
    copts = copts(),
)

cc_library(
    name = "rpc_error_code",
    hdrs = [
        "RpcErrorCode.h",
    ],
    deps = [
        "//rtp_llm/cpp/proto:model_rpc_service_cc_proto",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "py_utils",
    hdrs = [
        "PyUtils.h",
    ],
    srcs = [
        "PyUtils.cc",
    ],
    deps = torch_deps(),
    copts = copts(),
    visibility = ["//visibility:public"],
)
