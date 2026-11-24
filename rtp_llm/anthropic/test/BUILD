load("@arch_config//:arch_select.bzl", "telemetry_test_deps")

py_test(
    name = "anthropic_translator_test",
    srcs = ["anthropic_translator_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = [
        "//rtp_llm:anthropic_api",
        "//rtp_llm:testlib_lite",
    ],
)

py_test(
    name = "anthropic_frontend_test",
    srcs = ["anthropic_frontend_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = [
        "//rtp_llm:anthropic_api",
        "//rtp_llm:fastapi",
        "//rtp_llm:frontend",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm:testlib",
        "//rtp_llm:uvicorn",
    ] + telemetry_test_deps(),
)
