py_test(
    name = "startup_rank_selection_test",
    srcs = ["startup_rank_selection_test.py"],
    data = ["//rtp_llm:sdk"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "0"},
    timeout = "short",
)

py_test(
    name = "eplb_control_test",
    srcs = ["eplb_control_test.py"],
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "frontend_routing_test",
    srcs = ["frontend_routing_test.py"],
    deps = ["//rtp_llm:testlib"],
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
        "testdata/tokenizer",
        "testdata/cpt",
    ]),
)

py_test(
    name = "distributed_server_test",
    srcs = ["distributed_server_test.py"],
    data = [
        ":testdata",
        "//rtp_llm:sdk",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    timeout = 'short',
    env = {"GPU_COUNT": "0"},
)
