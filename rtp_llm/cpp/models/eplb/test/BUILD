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
        "//rtp_llm/cpp/pybind:th_compute_lib",
        "//rtp_llm/cpp/pybind:th_utils",
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
    name = "eplb_py_wrapper_test",
    srcs = [
        "eplb_py_wrapper_test.py",
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

# Pure-CPU unit test for the frozen EPLB window reducer (no GPU, no torch).
cc_test(
    name = "expert_load_metrics_test",
    srcs = [
        "ExpertLoadMetricsTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/models:expert_load_metrics",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# Real CUDA monitor lifecycle, including recovery after a rejected window.
cc_test(
    name = "expert_load_monitor_test",
    srcs = ["ExpertLoadMonitorTest.cc"],
    copts = copts(),
    # Link the real ExecOps implementation once in this standalone binary,
    # like core/test:exec_ops_test. Mixing the full compute shared library with
    # these static dependencies duplicates global constructors/destructors.
    deps = [
        "//rtp_llm/cpp/models:expert_load_monitor",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/cuda/ops:gpu_base",
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
    timeout = "short",
    target_compatible_with = select({
        "@//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
