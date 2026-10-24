load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "th_eplb_py_wrapper_test_lib",
    srcs = [
        "eplb_py_wrapper_test.cc",
    ],
    hdrs = [
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/models:eplb",
        "//rtp_llm/cpp/models:model_types",
        "//rtp_llm/cpp/pybind:th_utils",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_binary(
    name = "th_eplb_py_wrapper_test",
    deps = [":th_eplb_py_wrapper_test_lib"],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_test(
    name = "expert_balancer_step_test",
    srcs = [
        "expert_balancer_step_test.py",
        "eplb_runtime_test_utils.py",
    ],
    data = [":th_eplb_py_wrapper_test"],
    deps = ["//rtp_llm:torch"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    tags = ["L20_CU13"],
    timeout = "moderate",
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "eplb_py_wrapper_test",
    srcs = [
        "eplb_py_wrapper_test.py",
        "eplb_runtime_test_utils.py",
    ],
    data = [
        ":th_eplb_py_wrapper_test",
    ],
    deps = [
        "//rtp_llm:torch",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_test = cc_test_wrapper

cc_library(
    name = "expert_load_metrics_reference",
    testonly = True,
    srcs = ["ExpertLoadMetricsReference.cc"],
    hdrs = ["ExpertLoadMetricsReference.h"],
    copts = copts(),
    deps = ["//rtp_llm/cpp/models:expert_metric_utils"],
)

# CPU oracle tests for DeepInfer compact window reduction.
cc_test(
    name = "expert_load_metrics_test",
    testonly = True,
    srcs = [
        "ExpertLoadMetricsTest.cc",
    ],
    copts = copts(),
    deps = [
        ":expert_load_metrics_reference",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# Real CUDA monitor lifecycle and synchronous reporting.
cc_test(
    name = "expert_load_monitor_test",
    testonly = True,
    srcs = [
        "ExpertLoadMetricsCudaTest.cc",
        "ExpertLoadMonitorTest.cc",
        "ExpertLoadMonitorTestCommStub.cc",
    ],
    copts = copts(),
    deps = [
        ":expert_load_metrics_reference",
        "//rtp_llm/cpp/models:expert_metric_utils",
        "//rtp_llm/cpp/models:expert_load_monitor",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@rapidjson//:rapidjson",
    ] + torch_deps(),
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    tags = ["L20_CU13"],
    # GPU-lock and JIT-cache setup can take most of a "short" (60 s) budget
    # even though the test body runs in a few seconds.
    timeout = "moderate",
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
