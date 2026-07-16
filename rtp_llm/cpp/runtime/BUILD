load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

# Header-only — safe to link everywhere.
cc_library(
    name = "bootstrap_hdrs",
    hdrs = ["Bootstrap.h"],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ] + torch_deps(),
    copts = copts(),
)

# Impl + std::once_flag g_bootstrap_flag.  MUST live in exactly one .so so
# the flag is single-instance process-wide; double-instancing would let
# the second .so re-run the bootstrap on first call.
cc_library(
    name = "bootstrap_impl",
    srcs = ["Bootstrap.cc"],
    deps = [
        ":bootstrap_hdrs",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:stack_tracer",
    ] + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ],
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    copts = copts(),
    alwayslink = True,
)

alias(
    name = "bootstrap",
    actual = ":bootstrap_hdrs",
)

# Header-only interface: declarations of initRuntime / getDeviceId /
# getEnableCommOverlap / runtimeSyncAndCheck / runtimeCreateEvent / etc.  Safe
# to link into every .so.
cc_library(
    name = "cuda_runtime_hdrs",
    hdrs = ["CudaRuntime.h"],
    deps = [
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/core:device_data",
    ] + torch_deps(),
    copts = copts(),
)

# Implementation + the legacy globals (g_runtime_initialized, g_device_id,
# g_enable_comm_overlap).  MUST live in exactly one .so (librtp_compute_ops.so
# via th_compute_lib) so that `initRuntime` called from Python's init_exec_ctx
# and `getDeviceId` / `getEnableCommOverlap` called from the engine both see
# the same state.
cc_library(
    name = "cuda_runtime_impl",
    srcs = ["CudaRuntime.cc"],
    deps = [
        ":cuda_runtime_hdrs",
        ":bootstrap",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:device_pin",
        "//rtp_llm/cpp/model_utils:model_utils",
        "//rtp_llm/models_py/bindings/core:device_data",
    ] + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        ],
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    copts = copts(),
    alwayslink = True,
)

# Compat alias — header-only for generic callers.  th_compute_lib must depend
# on :cuda_runtime_impl explicitly.
alias(
    name = "cuda_runtime",
    actual = ":cuda_runtime_hdrs",
)
