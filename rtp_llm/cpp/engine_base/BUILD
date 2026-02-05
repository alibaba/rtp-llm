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
    deps = [
        "//:rtp_compute_ops",
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
        "//:rtp_compute_ops",
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
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

# Executor interface
cc_library(
    name = "executor",
    hdrs = [
        "Executor.h",
    ],
    deps = [
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/models:models",
        "//:rtp_compute_ops",
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
    deps = [
        ":executor",
        ":profiler",
        ":worker_status_info",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/models:lora",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/config:eplb_config",
        "//rtp_llm/cpp/utils:hash_util",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "arpc_server",
    hdrs = glob([
        "arpc/*",
    ]),
    srcs = glob([
        "arpc/*",
    ]),
    deps = [
        "@havenask//aios/network/arpc:arpc",
        "@havenask//aios/network/arpc/arpc/metric:kmonitor_anet_metric",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"]
)
