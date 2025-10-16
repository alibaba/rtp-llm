filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
    ]),
)

py_test(
    name = "multimodal_image_load_test",
    srcs = [
        "multimodal_image_load_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "multimodal_process_engine_test",
    srcs = [
        "multimodal_process_engine_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    data = [
        ":testdata",
    ],
    exec_properties = {'gpu':'A10'},
)