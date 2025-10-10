load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")


# Weights converter for model weight conversion
cc_library(
    name = "weights_converter",
    hdrs = [
        "WeightsConverter.h",
    ],
    srcs = [
        "WeightsConverter.cc",
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/config:gpt_init_params",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
    visibility = ["//visibility:public"],
)

# Load balancing and worker management
cc_library(
    name = "worker_status_info",
    hdrs = [
        "WorkerStatusInfo.h",
        "Host.h"
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

# Profiling and performance monitoring
cc_library(
    name = "profiler",
    hdrs = [
        "TorchProfiler.h",
    ],
    srcs = [
        "TorchProfiler.cc",
    ],
    deps = [
        "//rtp_llm/cpp/config:static_config",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

# Executor interface
cc_library(
    name = "executor",
    hdrs = [
        "Executor.h",
    ],
    deps = [
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/devices:devices_base",
        "@com_google_absl//absl/status",
    ],
    visibility = ["//visibility:public"],
)

# Main engine base class
cc_library(
    name = "engine_base",
    hdrs = [
        "EngineBase.h",
        "EngineInitParams.h",
        "ProposeModelEngineInitParams.h",
    ],
    srcs = [
        "EngineBase.cc",
    ],
    deps = torch_deps() + [
        ":executor",
        ":profiler",
        ":worker_status_info",
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/core:buffer",
        "//rtp_llm/cpp/config:gpt_init_params",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/models:eplb_config",
        "//rtp_llm/cpp/models:lora",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "@com_google_absl//absl/status",
    ],
    visibility = ["//visibility:public"],
)


