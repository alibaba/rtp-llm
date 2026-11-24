package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_qwen4_5_config_weight",
    srcs = ["test_qwen4_5_config_weight.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

test_suite(
    name = "qwen4_5_config_weight",
    tests = [
        ":test_qwen4_5_config_weight",
    ],
)
