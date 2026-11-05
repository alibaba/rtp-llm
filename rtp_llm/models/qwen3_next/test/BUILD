py_test(
    name = "qwen3_next_mtp_test",
    srcs = ["qwen3_next_mtp_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "qwen35_moe_text_test",
    srcs = ["qwen35_moe_text_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)
