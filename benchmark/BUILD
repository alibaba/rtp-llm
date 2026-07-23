load("@pip_cpu_torch//:requirements.bzl", requirement_cpu = "requirement")

py_test(
    name = "topology_kv_candidate_schedule_cpu_test",
    srcs = [
        "test_topology_kv_candidate_schedule.py",
        "topology_kv_candidate_schedule.py",
    ],
    main = "test_topology_kv_candidate_schedule.py",
    deps = select({
        "@rtp_llm//:using_rocm": [requirement_cpu("torch")],
        "//conditions:default": ["//rtp_llm:torch"],
    }),
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
