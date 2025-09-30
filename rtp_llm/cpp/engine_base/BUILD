load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "data_structures",
    hdrs = [
        "LoadBalance.h",
        "WorkerStatusInfo.h",
    ],
    srcs = [
        "LoadBalance.cc",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@havenask//aios/autil:json",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "engine_base",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = torch_deps() + [
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/models:lora",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
        "//rtp_llm/cpp/utils:signal_utils",
        "//rtp_llm/cpp/utils:hash_util",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@havenask//aios/autil:json",
    ],
    visibility = ["//visibility:public"],
)

