load("@arch_config//:arch_select.bzl", "requirement")
load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

requirement(["torch"])

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
