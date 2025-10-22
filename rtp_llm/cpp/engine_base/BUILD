load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

# Weights converter for model weight conversion
cc_library(
    name = "weights_converter",
    srcs = [
        "WeightsConverter.cc",
    ],
    hdrs = [
        "WeightsConverter.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/pybind:py_utils",
    ],
)

# Load balancing and worker management
cc_library(
    name = "worker_status_info",
    hdrs = [
        "Host.h",
        "WorkerStatusInfo.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

# Profiling and performance monitoring
cc_library(
    name = "profiler",
    srcs = [
        "TorchProfiler.cc",
    ],
    hdrs = [
        "TorchProfiler.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
    ],
)

# Executor interface
cc_library(
    name = "executor",
    hdrs = [
        "Executor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/engine_base/schedulers",
        "//rtp_llm/cpp/engine_base/stream",
        "//rtp_llm/cpp/models",
        "@com_google_absl//absl/status",
    ],
)

# Main engine base class
cc_library(
    name = "engine_base",
    srcs = [
        "EngineBase.cc",
    ],
    hdrs = [
        "EngineBase.h",
        "EngineInitParams.h",
        "ProposeModelEngineInitParams.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":executor",
        ":profiler",
        ":worker_status_info",
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/cache2",
        "//rtp_llm/cpp/engine_base/schedulers",
        "//rtp_llm/cpp/engine_base/stream",
        "//rtp_llm/cpp/engine_base/system_prompt",
        "//rtp_llm/cpp/models:lora",
        "//rtp_llm/cpp/utils:hash_util",
    ],
)
