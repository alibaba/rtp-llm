package(default_visibility = ["//visibility:public"])

py_library(
    name = "support",
    testonly = True,
    srcs = [
        "support/__init__.py",
        "support/fixtures.py",
        "support/gang.py",
        "support/plan.py",
        "support/reports.py",
        "support/results.py",
    ],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_entry_lib",
        "//rtp_llm/test/perf_test:perf_test_results_lib",
    ],
)

py_library(
    name = "runtime_support",
    testonly = True,
    srcs = ["support/execution.py", "support/prefill.py"],
    deps = [":support", "//rtp_llm/test/perf_test:perf_test_runtime_lib"],
)

# One target per functional test module; configuration variations use subTest.
_TESTS = {
    "aone/broker_test": ["//rtp_llm/test/perf_test:aone_lib"],
    "aone/gang_executor_test": ["//rtp_llm/test/perf_test:aone_lib"],
    "aone/gang_worker_test": ["//rtp_llm/test/perf_test:aone_lib"],
    "aone/plan_test": [],
    "aone/single_test": ["//rtp_llm/test/perf_test:aone_lib"],
    "aone/worker_test": ["//rtp_llm/test/perf_test:aone_lib"],
    "core/build_closure_test": [],
    "core/ci_test": ["//rtp_llm/test/perf_test:ci_helpers", "//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "core/config_test": [],
    "core/engine_paths_test": ["//rtp_llm/test/perf_test:aone_lib"],
    "core/launch_config_test": ["//rtp_llm/test/perf_test:aone_lib"],
    "core/local_entry_test": [],
    "core/entry_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "core/plan_test": [],
    "core/report_test": ["//rtp_llm/test/perf_test:perf_test_analyzer_lib", "//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "core/results_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "core/storage_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "decode/executor_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "decode/planner_test": [],
    "decode/remote_entry_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "distributed/decode_sync_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "distributed/lifecycle_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "distributed/prefill_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "execution/engine_session_test": ["//rtp_llm/test/perf_test:perf_test_lib"],
    "execution/local_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "execution/local_bazel_test": [],
    "execution/offline_runner_test": ["//rtp_llm/test/perf_test:perf_test_lib"],
    "execution/profiling_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "experiments/experiment_test": [],
    "policies/publication_test": ["//rtp_llm/test/perf_test:perf_test_publish_policy_lib"],
    "policies/regression_test": ["//rtp_llm/test/perf_test:perf_test_regression_lib"],
    "prefill/continuous_pressure_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "prefill/dashsc_transport_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "prefill/fingerprints_test": [],
    "prefill/formula_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "prefill/logarithmic_grid_test": ["//rtp_llm/test/perf_test:aone_lib"],
    "prefill/planner_test": [],
    "prefill/runner_test": ["//rtp_llm/test/perf_test:perf_test_runtime_lib"],
    "prefill/workload_test": [],
    "ssh/executor_test": ["//rtp_llm/test/perf_test:perf_test_ssh_lib"],
    "ssh/remote_test": ["//rtp_llm/test/perf_test:perf_test_ssh_lib"],
    "ssh/worker_test": ["//rtp_llm/test/perf_test:perf_test_ssh_lib"],
}

_RUNTIME_SUPPORT_TESTS = [
    "core/entry_test",
    "core/storage_test",
    "decode/executor_test",
    "distributed/prefill_test",
    "execution/engine_session_test",
    "execution/local_test",
    "execution/profiling_test",
    "experiments/experiment_test",
    "prefill/runner_test",
    "prefill/workload_test",
]

_EXTRA_SOURCES = {
    "core/local_entry_test": ["//rtp_llm/test/perf_test:local_main.py"],
    "execution/local_bazel_test": ["//rtp_llm/test/perf_test:local_main.py"],
    "core/build_closure_test": ["//rtp_llm/test/perf_test:main.py"],
    "core/entry_test": ["//rtp_llm/test/perf_test:main.py"],
    "decode/remote_entry_test": ["//rtp_llm/test/perf_test:decode_remote_test.py"],
}

_EXTRA_DATA = {
    "core/build_closure_test": [
        "//rtp_llm/test/perf_test:BUILD",
        "//rtp_llm/test/perf_test:perf_test_python_sources",
    ],
    "core/entry_test": ["//tools:perf_test_script"],
}

[
    py_test(
        name = module.replace("/", "_"),
        srcs = [module + ".py"] + _EXTRA_SOURCES.get(module, []),
        main = module + ".py",
        data = [
            "//rtp_llm/test/perf_test:aone_defs.bzl",
            "//rtp_llm/test/perf_test:perf_test_examples",
        ] + _EXTRA_DATA.get(module, []),
        deps = [":runtime_support" if module in _RUNTIME_SUPPORT_TESTS else ":support"] + deps,
    )
    for module, deps in _TESTS.items()
]

test_suite(
    name = "unit_tests",
    tests = [":" + module.replace("/", "_") for module in _TESTS],
)
