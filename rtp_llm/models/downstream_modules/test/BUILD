py_test(
    name = "jev_test",
    srcs = ["jev_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm:safetensors"],
    env = {"GPU_COUNT": "0"},
)

py_test(
    name = "kev_test",
    srcs = ["kev_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:safetensors",
    ],
    env = {"GPU_COUNT": "0"},
)
