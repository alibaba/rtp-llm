package(default_visibility = ["//visibility:public"])

py_library(
    name = "smoke_lib",
    srcs = [
        "checkpoint_guard.py",
        "grammar_smoke.py",
        "jit_smoke.py",
        "logprobs_smoke.py",
        "request_smoke.py",
        "result_check.py",
        "server_launcher.py",
        "text_smoke.py",
    ],
    # Preserve the scripts' sibling imports as well as their package imports.
    imports = ["."],
    deps = ["//rtp_llm:zstandard"],
)

py_library(
    name = "pd_test_support",
    srcs = ["pd_config.py", "worker.py"],
    deps = [":smoke_lib"],
)

py_library(
    name = "remote_pd_support",
    srcs = ["remote_pd_test.py"],
    deps = [":pd_test_support"],
)

_OFFLINE_TESTS = [
    "test_checkpoint_guard",
    "test_grammar_smoke",
    "test_jit_smoke",
    "test_logprobs_smoke",
    "test_orthogonal_smoke",
    "test_request_smoke",
    "test_remote_pd",
    "test_result_check",
    "test_server_launcher",
    "test_text_smoke_capped",
    "test_text_smoke_mtp",
    "test_worker",
]

[
    py_test(
        name = name,
        srcs = [name + ".py"],
        main = name + ".py",
        deps = [":pd_test_support", ":remote_pd_support", ":smoke_lib"],
        env = {"GPU_COUNT": "0"},
    )
    for name in _OFFLINE_TESTS
]

test_suite(
    name = "offline_tests",
    tests = [":" + name for name in _OFFLINE_TESTS],
)
