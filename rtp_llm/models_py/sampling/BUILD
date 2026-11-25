load("@arch_config//:arch_select.bzl", "triton_deps")


py_library(
    name = "sampling",
    srcs = [
        "__init__.py",
        "adapter.py",
        "draft_logits.py",
        "runtime.py",
        "rejection_metadata.py",
        "rejection_graph.py",
        "rejection_native.py",
        "rejection_compact.py",
        "vllm_gumbel.py",
        "vllm_rejection.py",
        "vllm_watermark.py",
        "vllm_philox.py",
        "vllm_prf_base.py",
        "vllm_fly.py",
    ],
    deps = ["//rtp_llm:numpy", "//rtp_llm:torch"] + triton_deps(["triton"]),
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_gumbel",
    srcs = ["test_gumbel.py"],
    deps = [":sampling"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "test_rejection",
    srcs = ["test_rejection.py"],
    deps = [":sampling"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "test_rejection_output_perf",
    srcs = ["test_rejection_output_perf.py"],
    deps = [":sampling"],
    tags = ["SM103_CU13", "manual"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "test_runtime",
    srcs = ["test_runtime.py"],
    deps = [":sampling"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "test_logits_filter",
    srcs = ["test_logits_filter.py", "logits_filter.py"],
    deps = ["//rtp_llm:torch"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)
