load("@arch_config//:arch_select.bzl", "torch_deps")

py_test(
    name = "topology_kv_candidate_schedule_test",
    srcs = [
        "test_topology_kv_candidate_schedule.py",
        "topology_kv_candidate_schedule.py",
    ],
    main = "test_topology_kv_candidate_schedule.py",
    deps = torch_deps(),
)

py_test(
    name = "topology_kv_policy_test",
    srcs = ["test_topology_kv_policy.py"],
    data = ["//rtp_llm/models_py:topology_kv_policy_runfiles"],
    main = "test_topology_kv_policy.py",
    deps = torch_deps(),
)

py_test(
    name = "topology_kv_candidate_schedule_cuda_manual_test",
    srcs = [
        "test_topology_kv_candidate_schedule.py",
        "topology_kv_candidate_schedule.py",
    ],
    args = ["-k", "cuda"],
    env = {"RTP_LLM_RUN_MANUAL_BENCHMARK_TESTS": "1"},
    main = "test_topology_kv_candidate_schedule.py",
    tags = ["manual"],
    deps = torch_deps(),
)
