py_test(
    name = "topology_kv_candidate_schedule_cpu_test",
    srcs = [
        "test_topology_kv_candidate_schedule.py",
        "topology_kv_candidate_schedule.py",
    ],
    main = "test_topology_kv_candidate_schedule.py",
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "topology_kv_candidate_schedule_cuda_test",
    srcs = [
        "test_topology_kv_candidate_schedule_cuda.py",
        "topology_kv_candidate_schedule.py",
    ],
    main = "test_topology_kv_candidate_schedule_cuda.py",
    tags = ["manual"],
    deps = ["//rtp_llm:torch"],
)
