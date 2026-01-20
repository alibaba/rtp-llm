py_library(
    name = "perf_test_lib",
    srcs = [
        "batch_decode_test.py",
        "batch_perf_impl.py",
        "dataclass.py",
        "test_util.py",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:device_resource",
    ],
    visibility = ["//visibility:public"],
)

py_test(
        name = "perf_test",
        main = "test_entry.py",
        srcs = [
            "test_entry.py",
        ],
        timeout = "eternal",
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/test/perf_test:perf_test_lib",
        ],
        data = [
            "//rtp_llm:sdk",
        ],
        tags = ["manual"],
    )