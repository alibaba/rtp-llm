load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

# Header-only role classification for the startup warmup. Deliberately kept to one dependency
# (config:role_types) so warmup_role_gate_test runs host-side with no torch and no GPU: the
# classification decides how much KV cache a whole role class allocates, so it needs gated coverage
# that does not require standing up an engine. This target is the header's single owner;
# normal_engine depends on it explicitly below.
cc_library(
    name = "warmup_role_gate",
    hdrs = ["WarmupRoleGate.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/config:role_types",
    ],
)

cc_library(
    name = "normal_engine",
    hdrs = glob([
        "*.h",
        "speculative/*.h",
    ], exclude = ["WarmupRoleGate.h"]),
    srcs = glob([
        "*.cc",
        "speculative/*.cc",
    ]),
    deps = [
        ":warmup_role_gate",
        "//rtp_llm/cpp/cache:warmup_result_assembly",
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/models:logits_processor",
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/utils:device_pin",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base:worker_status_info",
        "//rtp_llm/cpp/engine_base:profiler",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt_constructor",
        "//rtp_llm/cpp/models:eplb",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/utils:tensor_debug_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:standalone_ops",
            "//rtp_llm/models_py/bindings/cuda/kernels:mtp_target_verify_prepare",
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "@//:using_rocm": [
            "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
