load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "cuda13_test_exec_properties", "torch_deps")
load("@rules_cc//cc:defs.bzl", cuda13_cc_test = "cc_test")

cuda13_cc_test(
    name = "cuda_graph_replay_retry_integration_test",
    srcs = ["cuda_graph_replay_retry_integration_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/utils:torch_cuda_oom",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = cuda13_test_exec_properties(),
    target_compatible_with = select({
        "//:using_cuda13_arm": [],
        "//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    timeout = "short",
)

cc_test(
    name = "combo_position_ids_validation_test",
    srcs = ["combo_position_ids_validation_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:combo_position_ids_validation",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "cuda_graph_copy_kernel_test_rocm",
    main = "cuda_graph_copy_kernel_test.py",
    srcs = ["cuda_graph_copy_kernel_test.py"],
    data = ["//:th_transformer"],
    deps = ["//rtp_llm/test/model_test/test_util:test_util"],
    env = {
        "NOT_USE_DEFAULT_STREAM": "1",
        "TEST_USING_DEVICE": "ROCM",
        "ENABLE_CUDA_GRAPH_DEBUG_MODE": "1",
    },
    tags = ["manual", "rocm"],  # Collected automatically by py_ut_amd.
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

cc_test(
    name = "prepared_attention_inputs_guard_test",
    srcs = ["prepared_attention_inputs_guard_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:prepared_attention_inputs_guard",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "generation_prefill_cuda_graph_replay_metadata_test",
    srcs = ["generation_prefill_cuda_graph_replay_metadata_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:generation_prefill_cuda_graph_replay_metadata",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cuda_graph_prepare_fill_rocm_test",
    srcs = ["cuda_graph_prepare_fill_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/cuda/kernels:cuda_graph_prepare",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
        "@com_google_googletest//:gtest_main",
    ],
    tags = ["rocm"],
    target_compatible_with = select({
        "@//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    exec_properties = {"gpu": "MI308X-ROCM7"},
)

cc_library(
    name = "test_cuda_graph_runner_libs",
    srcs = [
        "cuda_graph_test_runner.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_binary(
    name = "test_cuda_graph_runner",
    deps = [
        ":test_cuda_graph_runner_libs",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

# The Python drivers are collected by pytest's H20 profile. Python packages are
# installed by pyproject/uv and are intentionally no longer modeled as Bazel
# dependencies. Keep only the compiled binding here for setup.py to stage.
