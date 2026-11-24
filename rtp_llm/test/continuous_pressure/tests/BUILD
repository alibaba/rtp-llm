py_test(
    name = "core_test",
    srcs = ["core_test.py"],
    deps = ["//rtp_llm/test/continuous_pressure"],
)

py_test(
    name = "evidence_test",
    srcs = [
        "__init__.py",
        "core_test.py",
        "evidence_test.py",
    ],
    deps = ["//rtp_llm/test/continuous_pressure"],
)

py_test(
    name = "independence_test",
    srcs = ["independence_test.py"],
    deps = ["//rtp_llm/test/continuous_pressure"],
)

test_suite(
    name = "unit_tests",
    tests = [
        ":adapter_test",
        ":capture_test",
        ":rpc_wire_test",
        ":core_test",
        ":evidence_test",
        ":independence_test",
    ],
)


py_test(
    name = "adapter_test",
    srcs = ["adapter_test.py"],
    deps = ["//rtp_llm/test/continuous_pressure:rtp_adapter"],
)

py_test(
    name = "rpc_wire_test",
    srcs = ["rpc_wire_test.py"],
    data = ["//rtp_llm/libs:libth_transformer_config_so"],
    deps = ["//rtp_llm/test/continuous_pressure:rtp_sdk"],
)

py_test(
    name = "capture_test",
    srcs = ["capture_test.py"],
    deps = ["//rtp_llm/test/continuous_pressure:rtp_adapter"],
)
