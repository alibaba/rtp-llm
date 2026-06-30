load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "normal_engine",
    hdrs = glob([
        "*.h",
        "speculative/*.h",
    ]),
    # speculative/*.cc is GPU-only (MTP kernels, CUDA tensors). Compile it
    # only on cuda/rocm; CPU/ARM consumers of normal_engine never hit MTP paths.
    srcs = glob(["*.cc"]) + select({
        "//:using_cuda": glob(["speculative/*.cc"]),
        "@//:using_rocm": glob(["speculative/*.cc"]),
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm/cpp/models:models",
        "//rtp_llm/cpp/models:logits_processor",
        "//rtp_llm/cpp/engine_base:engine_base",
        "//rtp_llm/cpp/models:grammar_logits_processor",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/engine_base:worker_status_info",
        "//rtp_llm/cpp/engine_base:profiler",
        "//rtp_llm/cpp/engine_base/schedulers:schedulers",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt_constructor",
        "//rtp_llm/cpp/models:eplb",
        "//rtp_llm/cpp/utils:tensor_debug_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:standalone_ops",
        ],
        "@//:using_rocm": [],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
