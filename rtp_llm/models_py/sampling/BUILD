load("//bazel:arch_select.bzl", "requirement")

requirement(["numpy", "torch", "triton"])

py_library(
    name = "sampling",
    srcs = [
        "__init__.py",
        "adapter.py",
        "acceptance_estimator.py",
        "adaptive_controller.py",
        "adaptive_runtime.py",
        "adaptive_session.py",
        "adaptive_verification.py",
        "draft_logits.py",
        "runtime.py",
        "rejection_metadata.py",
        "rejection_graph.py",
        "rejection_native.py",
        "vllm_gumbel.py",
        "vllm_rejection.py",
        "vllm_watermark.py",
        "vllm_philox.py",
        "vllm_prf_base.py",
        "vllm_fly.py",
    ],
    deps = [":numpy", ":torch", ":triton"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_adaptive_controller",
    srcs = ["test_adaptive_controller.py"],
    deps = [":sampling"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "test_adaptive_runtime",
    srcs = ["test_adaptive_runtime.py", "adaptive_runtime.py"],
)

py_test(
    name = "test_adaptive_verification",
    srcs = ["test_adaptive_verification.py"],
    deps = [":sampling"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)

py_test(
    name = "test_acceptance_estimator",
    srcs = ["test_acceptance_estimator.py"],
    deps = [":sampling"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
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
    deps = [":torch"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
)
