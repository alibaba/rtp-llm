package(default_visibility = ["//visibility:public"])

load("@rules_python//experimental/python:wheel.bzl", "py_package")
load("//rtp_llm/flexlb/navi_sched/tools:mangled_wheel.bzl", "mangled_py_wheel")

sh_binary(
    name = "python310_wheelmaker",
    srcs = ["python310_wheelmaker.sh"],
    data = ["@rules_python//experimental/tools:wheelmaker"],
)

py_package(
    name = "navi_sched_py_pkg",
    deps = ["//rtp_llm/flexlb/navi_sched:py_sched_lib"],
)

mangled_py_wheel(
    name = "navi_sched_wheel",
    distribution = "navi_sched",
    version = "0.1.0",
    deps = [":navi_sched_py_pkg"],
    entry_points = {
        "console_scripts": [
            "navi_sched_replay = navi_sched.replay_cli:main",
        ],
        "vllm.general_plugins": [
            "navi_sched = navi_sched_vllm_plugin:register_vllm_plugin",
        ],
    },
    strip_path_prefixes = ["rtp_llm/flexlb/navi_sched"],
)

py_binary(
    name = "mangle_and_flatten",
    srcs = ["mangle_and_flatten.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "mangle_and_flatten_test",
    srcs = ["mangle_and_flatten_test.py"],
    data = ["mangle_and_flatten.py"],
    python_version = "PY3",
)

py_test(
    name = "analyze_queue_timing_test",
    srcs = [
        "analyze_queue_timing.py",
        "analyze_queue_timing_test.py",
    ],
    main = "analyze_queue_timing_test.py",
    python_version = "PY3",
)

py_test(
    name = "analyze_pending_cache_keys_test",
    srcs = [
        "analyze_pending_cache_keys.py",
        "analyze_pending_cache_keys_test.py",
    ],
    main = "analyze_pending_cache_keys_test.py",
    python_version = "PY3",
)
