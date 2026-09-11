py_test(
    name = "deepseek_v41_processor_test",
    srcs = ["deepseek_v41_processor_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["requires-dsv41-weights"],
)

py_test(
    name = "deepseek_v41_vision_test",
    srcs = ["deepseek_v41_vision_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["requires-dsv41-weights"],
)

py_test(
    name = "deepseek_vision_test",
    srcs = ["deepseek_vision_test.py"],
    deps = ["//rtp_llm:testlib"],
    tags = ["requires-dsv41-weights"],
)

py_test(
    name = "multimodal_image_load_test",
    srcs = [
        "multimodal_image_load_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)
