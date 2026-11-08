package(default_visibility = ["//visibility:public"])

# CI only owns rendezvous, resources and evidence transport. Requests and
# acceptance checks come from the traditional smoke package.
py_library(
    name = "smoke_lib",
    srcs = glob(["*.py"], exclude = ["test_*.py"]),
    deps = [
        "//rtp_llm/test/smoke/kimi_k3:smoke_lib",
        "//rtp_llm/test/smoke/kimi_k3:pd_test_support",
        "//rtp_llm/test/utils:coredump_util",
    ],
)

_TESTS = [
    "test_smoke_flow",
    "test_core_evidence",
    "test_k31_ci",
    "test_long_ci",
    "test_remote_pd",
    "test_shared_smoke",
]

[
    py_test(
        name = name,
        srcs = [name + ".py"],
        main = name + ".py",
        data = ["//rtp_llm:config/log_config.py"] if name == "test_server_launcher" else [],
        deps = [":smoke_lib"],
        tags = ["cpu"],
        env = {"GPU_COUNT": "0"},
    )
    for name in _TESTS
]

test_suite(name = "offline_tests", tests = [":" + name for name in _TESTS])
