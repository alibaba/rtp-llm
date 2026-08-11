load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "device_status",
    hdrs = ["DeviceStatus.h"],
    copts = copts(),
)

cc_library(
    name = "device_types",
    hdrs = ["DeviceTypes.h"],
    deps = select({
        "@//:using_cuda": ["@local_config_cuda//cuda:cuda_headers"],
        "@//:using_rocm": ["@local_config_rocm//rocm:rocm_headers"],
        "//conditions:default": [],
    }),
    copts = copts(),
)

cc_library(
    name = "rocm_compat",
    hdrs = [
        "rocm/AmdBfloat16.h",
        "rocm/CudaShims.h",
    ],
    deps = select({
        "@//:using_rocm": [
            "@local_config_rocm//rocm:hip",
            "@local_config_rocm//rocm:rocm",
            "@local_config_rocm//rocm:rocm_headers",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
)

cc_library(
    name = "device_error",
    srcs = ["DeviceError.cc"],
    hdrs = ["DeviceError.h"],
    deps = [
        ":device_types",
        "//rtp_llm/cpp/utils:core_utils",
    ] + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "@//:using_rocm": [
            "@local_config_rocm//rocm:hip",
            "@local_config_rocm//rocm:rocm_headers",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
)

# Header-only interface: declarations of initRuntime / getDeviceId /
# getEnableCommOverlap / runtimeSyncAndCheck / runtimeCreateEvent / etc.  Safe
# to link into every .so.
cc_library(
    name = "cuda_runtime_hdrs",
    hdrs = ["CudaRuntime.h"],
    deps = [
        ":device_status",
        "//rtp_llm/cpp/model_utils:model_utils",
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
        ":device_error",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:device_pin",
        "//rtp_llm/cpp/model_utils:model_utils",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/autil:stack_tracer",
    ] + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:rocm",
        ],
        "//conditions:default": [],
    }) + torch_deps(),
    copts = copts(),
    alwayslink = True,
    visibility = [
        "//rtp_llm/cpp/comm/test:__pkg__",
        "//rtp_llm/cpp/pybind:__pkg__",
        "//rtp_llm/models_py/bindings/core:__pkg__",
    ],
)
