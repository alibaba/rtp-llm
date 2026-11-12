load("//bazel:arch_select.bzl", "requirement")
load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

requirement(["torch"])



py_test(
    name = "deepep_utils_test",
    srcs = ["deepep_utils_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)


cuda13_sm100_py_test(
    name = "glm5_gate_pack_test",
    srcs = ["glm5_gate_pack_test.py"],
    timeout = "long",
    deps = [":test_support", "//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

cuda13_sm100_py_test(
    name = "glm5_mega_gate_test",
    srcs = ["glm5_mega_gate_test.py"],
    timeout = "long",
    deps = [":test_support", "//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

cuda13_sm100_py_test(
    name = "glm5_routing_fusion_test",
    srcs = ["glm5_routing_fusion_test.py"],
    timeout = "long",
    deps = [
        ":test_support",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

cuda13_sm100_py_test(
    name = "glm53_pd_long_eplb_test",
    srcs = ["glm53_pd_long_eplb_test.py", "glm53_pd_eplb_test.py"],
    gpu_count = 6,
    tags = ["manual"],
    timeout = "eternal",
    deps = ["//rtp_llm:testlib", "//rtp_llm/test/utils:maga_server_manager", "//rtp_llm/test/smoke:eplb_smoke_checks"],
    data = ["//rtp_llm:sdk"],
)

cuda13_sm100_py_test(
    name = "glm53_pd_eplb_test",
    srcs = ["glm53_pd_eplb_test.py"],
    gpu_count = 6,
    tags = ["manual"],
    timeout = "eternal",
    deps = ["//rtp_llm:testlib", "//rtp_llm/test/utils:maga_server_manager", "//rtp_llm/test/smoke:eplb_smoke_checks"],
    data = ["//rtp_llm:sdk"],
)





cuda13_sm100_py_test(
    name = "glm53_service_perf_test",
    main = "glm53_service_perf_test.py",
    srcs = ["glm53_service_perf_test.py"],
    gpu_count = 4,
    tags = ["manual"],
    timeout = "eternal",
    deps = ["//rtp_llm:testlib", "//rtp_llm/test/utils:maga_server_manager"],
    data = ["//rtp_llm:sdk"],
)

cuda13_sm100_py_test(
    name = "glm53_four_gpu_eplb_test",
    main = "glm53_server_eplb_test.py",
    srcs = ["glm53_server_eplb_test.py"],
    gpu_count = 4,
    tags = ["manual"],
    timeout = "eternal",
    env = {
        "EPLB_GLM5_WORLD": "4",
        "EPLB_GLM5_TP": "1",
        "EPLB_GLM5_REDUNDANT": "16",
        "EPLB_GLM5_MODES": "STATS,ALL",
    },
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/smoke:eplb_smoke_checks",
    ],
    data = ["//rtp_llm:sdk"],
)

cuda13_sm100_py_test(
    name = "glm53_server_eplb_test",
    srcs = ["glm53_server_eplb_test.py"],
    gpu_count = 8,
    tags = ["manual"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/smoke:eplb_smoke_checks",
    ],
    data = ["//rtp_llm:sdk"],
)

py_library(
    name = "test_support",
    srcs = ["test_support.py"],
    deps = ["//rtp_llm:eplb", ":torch"],
)

py_test(
    name = "glm5_wrapper_test",
    srcs = ["glm5_wrapper_test.py"],
    deps = [
        ":test_support",
        "//rtp_llm:eplb",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

cuda13_sm100_py_test(
    name = "glm5_mega_distributed_test",
    srcs = ["glm5_mega_distributed_test.py"],
    timeout = "long",
    gpu_count = 2,
    env = {"GPU_COUNT": "2"},
    deps = [
        ":test_support",
        "//rtp_llm:eplb",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "policy_test",
    srcs = ["policy_test.py"],
    data = ["policy_golden.json"],
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "policy_gpu_test",
    main = "policy_test.py",
    srcs = ["policy_test.py"],
    data = ["policy_golden.json"],
    env = {"EPLB_POLICY_TEST_DEVICE": "cuda"},
    deps = ["//rtp_llm:eplb", ":torch"],
)

py_test(
    name = "shuffle_test",
    srcs = ["shuffle_test.py"],
    deps = ["//rtp_llm:eplb", ":torch"],
)

py_test(
    name = "policy_fused_test",
    srcs = ["policy_fused_test.py", "policy_reference.py"],
    data = ["policy_replay.json"],
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "policy_fused_gpu_test",
    main = "policy_fused_test.py",
    srcs = ["policy_fused_test.py", "policy_reference.py"],
    data = ["policy_replay.json"],
    env = {"EPLB_POLICY_TEST_DEVICE": "cuda"},
    timeout = "long",
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "runtime_distributed_test",
    srcs = ["runtime_distributed_test.py", "runtime_test.py"],
    gpu_count = 2,
    timeout = "moderate",
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "shuffle_large_test",
    srcs = ["shuffle_large_test.py"],
    gpu_count = 4,
    tags = ["manual"],
    timeout = "long",
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "shuffle_large_8gpu_test",
    main = "shuffle_large_test.py",
    srcs = ["shuffle_large_test.py"],
    gpu_count = 8,
    env = {
        "EPLB_TEST_WORLD_SIZE": "8",
        "EPLB_TEST_LAYERS": "75",
        "EPLB_TEST_UPDATES": "10",
    },
    tags = ["manual"],
    timeout = "long",
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "runtime_distributed_8gpu_test",
    main = "runtime_distributed_test.py",
    srcs = ["runtime_distributed_test.py", "runtime_test.py"],
    gpu_count = 8,
    env = {"EPLB_TEST_WORLD_SIZE": "8"},
    tags = ["manual"],
    timeout = "moderate",
    deps = ["//rtp_llm:eplb", ":torch"],
)

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    deps = ["//rtp_llm:eplb", ":torch"],
)

py_test(
    name = "policy_larges_test",
    srcs = ["policy_larges_test.py", "policy_larges_reference.py", "policy_reference.py"],
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "policy_larges_gpu_test",
    main = "policy_larges_test.py",
    srcs = ["policy_larges_test.py", "policy_larges_reference.py", "policy_reference.py"],
    env = {"EPLB_POLICY_TEST_DEVICE": "cuda"},
    timeout = "long",
    deps = ["//rtp_llm:eplb", ":torch"],
)

cuda13_sm100_py_test(
    name = "policy_larges_distributed_8gpu_test",
    main = "policy_larges_distributed_test.py",
    srcs = [
        "policy_larges_distributed_test.py",
        "policy_larges_test.py",
        "policy_larges_reference.py",
        "policy_reference.py",
        "runtime_test.py",
    ],
    gpu_count = 8,
    tags = ["manual"],
    timeout = "long",
    deps = ["//rtp_llm:eplb", ":torch"],
)

py_test(
    name = "model_binding_test",
    srcs = ["model_binding_test.py"],
    deps = ["//rtp_llm/models_py:utils", ":torch"],
)
