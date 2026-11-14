package(default_visibility = ["//visibility:public"])

load(":aone_defs.bzl", "unified_decode_remote_test")

load("@pip_gpu_cuda13_torch//:requirements.bzl", requirement_cuda13 = "requirement")
load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_cuda12 = "requirement")
load("@pip_gpu_cuda12_9_torch//:requirements.bzl", requirement_cuda12_9 = "requirement")
load("@pip_cuda12_arm_torch//:requirements.bzl", requirement_cuda12_arm = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", requirement_cuda13_arm = "requirement")
load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_rocm = "requirement")
load("@pip_cpu_torch//:requirements.bzl", requirement_cpu = "requirement")
load("@pip_arm_torch//:requirements.bzl", requirement_arm = "requirement")
load("@pip_ppu_torch//:requirements.bzl", requirement_ppu = "requirement")

config_setting(
    name = "using_cuda12_x86_pre_12_9",
    define_values = {
        "using_cuda12_x86": "true",
        "using_cuda12_9_x86": "false",
    },
)

py_library(
    name = "perf_test_entry_lib",
    srcs = [
        "aone/__init__.py",
        "aone/config_schema.py",
        "aone/plan.py",
        "aone/sharding.py",
        "capabilities.py",
        "cli.py",
        "config_resolver.py",
        "config_schema.py",
        "decode/__init__.py",
        "decode/distribution.py",
        "decode/metrics.py",
        "decode/model.py",
        "decode/planner.py",
        "decode/scheduler.py",
        "entry.py",
        "case_planning.py",
        "errors.py",
        "plan.py",
        "prefill/__init__.py",
        "prefill/block_size.py",
        "prefill/case_store.py",
        "prefill/fingerprints.py",
        "prefill/measurement.py",
        "prefill/planner.py",
        "prefill/prompts.py",
        "prefill/random_batch.py",
        "prefill/traffic_batch.py",
        "prefill/traffic_distribution.py",
        "prefill/generate_traffic_batch_grid.py",
        "prefill/generate_random_batch_grid.py",
        "run_directory.py",
        "state.py",
        "storage.py",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "perf_test_results_lib",
    srcs = ["results.py"],
    deps = [":perf_test_entry_lib"],
)

py_library(
    name = "perf_test_runtime_lib",
    srcs = [
        "decode/executor.py",
        "decode/finalize.py",
        "decode/profiling.py",
        "decode/report.py",
        "decode/strategy.py",
        "decode/transport.py",
        "execution/engine_session.py",
        "execution/local.py",
        "execution/runtime.py",
        "execution/scheduler.py",
        "modes.py",
        "multi_node/decode_runtime.py",
        "multi_node/decode_sync.py",
        "multi_node/multi_benchmark.py",
        "multi_node/gang_config.py",
        "multi_node/local_server_runner.py",
        "multi_node/prefill_runtime.py",
        "prefill/checkpoint.py",
        "prefill/chart.py",
        "prefill/finalize.py",
        "prefill/dashsc_transport.py",
        "prefill/formula.py",
        "prefill/profile.py",
        "prefill/probe.py",
        "prefill/report.py",
        "prefill/restricted_symbolic.py",
        "prefill/runner.py",
        "prefill/trace.py",
        "test_util.py",
    ],
    deps = [
        ":perf_test_entry_lib",
        ":perf_test_results_lib",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
    ] + select({
        "@rtp_llm//:using_cuda13_x86": [requirement_cuda13("plotly")],
        "@rtp_llm//:using_cuda13_arm": [requirement_cuda13_arm("plotly")],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

exports_files([
    "aone/gang_worker.py",
    "aone/worker.py",
    "aone_defs.bzl",
    "decode_remote_test.py",
])

py_library(
    name = "aone_lib",
    srcs = [
        "aone/broker.py",
        "aone/gang_worker.py",
        "aone/collector.py",
        "aone/package.py",
        "aone/worker.py",
        "execution/aone.py",
        "execution/aone_bazel.py",
        "execution/aone_gang.py",
        "hub_download.py",
    ],
    deps = [
        ":perf_test_entry_lib",
        ":perf_test_runtime_lib",
    ],
)

# Optional, portable result policies. Normal perf_test execution does not
# import these libraries or run baseline comparisons and publishers implicitly.
py_library(
    name = "perf_test_regression_lib",
    srcs = [
        "regression/__init__.py",
        "regression/baseline.py",
        "regression/golden.py",
    ],
    deps = [":perf_test_results_lib"],
)

py_library(
    name = "perf_test_publish_policy_lib",
    srcs = [
        "publish/__init__.py",
        "publish/policy.py",
    ],
    deps = [":perf_test_entry_lib"],
)

py_library(
    name = "perf_test_regression_test_support",
    testonly = True,
    srcs = ["regression/test_support.py"],
    deps = [":perf_test_regression_lib"],
)

py_binary(
    name = "generate_perf_test_golden",
    srcs = ["generate_perf_test_golden.py"],
    deps = [":perf_test_regression_lib"],
)

py_binary(
    name = "validate_perf_test_golden",
    srcs = ["validate_perf_test_golden.py"],
    deps = [":perf_test_regression_lib"],
)

py_library(
    name = "perf_test_lib",
    srcs = [
        "multi_node/decode_runtime.py",
        "multi_node/decode_sync.py",
        "multi_node/gang_config.py",
        "multi_node/local_server_runner.py",
        "multi_node/prefill_runtime.py",
        "aone/__init__.py",
        "aone/config_schema.py",
        "aone/plan.py",
        "aone/sharding.py",
        "capabilities.py",
        "cli.py",
        "config_resolver.py",
        "config_schema.py",
        "dataset.py",
        "decode/executor.py",
        "decode/profiling.py",
        "decode/report.py",
        "decode/strategy.py",
        "decode/transport.py",
        "decode/distribution.py",
        "decode/finalize.py",
        "decode/planner.py",
        "decode/metrics.py",
        "decode/model.py",
        "decode/scheduler.py",
        "entry.py",
        "case_planning.py",
        "errors.py",
        "execution/engine_session.py",
        "execution/local.py",
        "execution/runtime.py",
        "execution/scheduler.py",
        "engine_paths.py",
        "hub_download.py",
        "offline_bench_test.py",
        "offline_runner.py",
        "plan.py",
        "prefill/block_size.py",
        "prefill/case_store.py",
        "prefill/fingerprints.py",
        "prefill/measurement.py",
        "prefill/planner.py",
        "prefill/random_batch.py",
        "prefill/traffic_batch.py",
        "prefill/traffic_distribution.py",
        "prefill/generate_traffic_batch_grid.py",
        "prefill/generate_random_batch_grid.py",
        "prefill/checkpoint.py",
        "prefill/chart.py",
        "prefill/finalize.py",
        "prefill/dashsc_transport.py",
        "prefill/formula.py",
        "prefill/profile.py",
        "prefill/probe.py",
        "prefill/report.py",
        "prefill/restricted_symbolic.py",
        "prefill/runner.py",
        "prefill/trace.py",
        "server.py",
        "run_directory.py",
        "storage.py",
        "state.py",
        "modes.py",
        "test_util.py",
    ],
    data = glob(["examples/*.json"]),
    deps = [
        ":aone_lib",
        ":perf_test_results_lib",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:device_resource",
    ] + select({
        "@rtp_llm//:using_cuda13_x86": [requirement_cuda13("plotly")],
        "@rtp_llm//:using_cuda13_arm": [requirement_cuda13_arm("plotly")],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "perf_test",
    main = "main.py",
    srcs = ["main.py"],
    deps = [
        ":aone_lib",
        ":perf_test_entry_lib",
        ":perf_test_runtime_lib",
    ],
)

# Offline throughput benchmark (new entry point).
# Manual target: Qwen3.5 weights must be available locally or resolvable by perf path resolution.
py_test(
    name = "offline_bench_test",
    main = "offline_bench_test.py",
    srcs = ["offline_bench_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = [
        "//rtp_llm:sdk",
    ],
    args = [
        "--checkpoint_path", "Qwen/Qwen3.5-35B-A3B",
        "--model_type", "qwen35_moe",
        "--input_len_min", "128",
        "--input_len_max", "512",
        "--output_len_min", "64",
        "--output_len_max", "256",
        "--concurrency_limit", "256",
        "--seq_size_per_block", "1024",
    ],
    tags = ["manual"],
)

py_binary(
    name = "generate_random_batch_grid",
    main = "prefill/generate_random_batch_grid.py",
    srcs = ["prefill/generate_random_batch_grid.py"],
    deps = [":perf_test_entry_lib"],
)

py_binary(
    name = "generate_traffic_batch_grid",
    main = "prefill/generate_traffic_batch_grid.py",
    srcs = ["prefill/generate_traffic_batch_grid.py"],
    deps = [":perf_test_entry_lib"],
)

py_library(
    name = "perf_test_support",
    testonly = True,
    srcs = [
        "test_support.py",
        "multi_node/analyzer/batch_analyze.py",
        "multi_node/analyzer/merge_decode_results.py",
        "multi_node/analyzer/trace_analyze.py",
    ],
    deps = [":perf_test_runtime_lib"],
)

py_test(
    name = "decode_executor_test",
    srcs = ["decode_executor_test.py"],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "decode_multi_node_test",
    srcs = ["decode_multi_node_test.py"],
    data = [
        "multi_node/multi_benchmark_config.yaml",
        "multi_node/gang_config.py",
        "multi_node/local_server_runner.py",
        "multi_node/multi_local_executor.sh",
        "multi_node/multi_runner.sh",
    ] + glob(["multi_node/analyzer/configs/*.py"]),
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "engine_session_test",
    srcs = ["engine_session_test.py"],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "entry_test",
    srcs = ["entry_test.py", "main.py"],
    data = [
        "//tools:perf_test_script",
        "examples/prefill_local.json",
        "examples/decode_grid_local.json",
        "testdata/historical_aone_plan.json",
        "BUILD",
    ] + glob(["**/*.py"]),
    deps = [":perf_test_support", ":perf_test_entry_lib"],
)

py_test(
    name = "offline_runner_test",
    srcs = ["offline_runner_test.py"],
    deps = [":perf_test_lib"],
)

py_test(
    name = "plan_test",
    srcs = ["plan_test.py"],
    data = [
        "testdata/historical_aone_plan.json",
        "examples/prefill_local.json",
        "examples/decode_grid_local.json",
    ],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "prefill_profile_test",
    srcs = ["prefill_profile_test.py"],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "prefill_report_test",
    srcs = ["prefill_report_test.py"],
    data = glob(["multi_node/analyzer/configs/*.py"]),
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "prefill_resume_test",
    srcs = ["prefill_resume_test.py"],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "prefill_runner_test",
    srcs = ["prefill_runner_test.py"],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "result_envelope_test",
    srcs = ["result_envelope_test.py"],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "runtime_adapter_test",
    srcs = ["runtime_adapter_test.py"],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "ssh_prefill_test",
    srcs = ["ssh_prefill_test.py"],
    data = [
        "multi_node/multi_benchmark_config.yaml",
        "multi_node/gang_config.py",
        "multi_node/local_server_runner.py",
        "multi_node/multi_local_executor.sh",
        "multi_node/multi_runner.sh",
    ],
    deps = [":perf_test_support", ":perf_test_lib"],
)

py_test(
    name = "unified_config_test",
    srcs = ["unified_config_test.py"],
    deps = [":perf_test_support", ":perf_test_entry_lib"],
)

py_test(
    name = "workload_test",
    srcs = ["workload_test.py"],
    data = [
        "examples/prefill_random_batch_local.json",
        "examples/req_distribution.json",
    ],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "aone_single_test",
    srcs = ["aone_single_test.py"],
    data = ["aone_defs.bzl"],
    deps = [":aone_lib", ":perf_test_support"],
)

py_test(
    name = "aone_worker_test",
    srcs = ["aone_worker_test.py"],
    deps = [":aone_lib", ":perf_test_support"],
)

py_test(
    name = "aone_broker_test",
    srcs = ["aone_broker_test.py"],
    data = ["aone_defs.bzl"],
    deps = [":aone_lib", ":perf_test_support"],
)

py_test(
    name = "aone_gang_test",
    srcs = ["aone_gang_test.py"],
    data = ["aone_defs.bzl"] + glob(["examples/*.json"]),
    deps = [":aone_lib", ":perf_test_support"],
)

# One test action runs the complete local-backend grid on a worker with two GPUs.
# Adapt the example model/tokenizer paths to files readable on that worker.
unified_decode_remote_test(
    name = "decode_remote_example",
    config = "examples/decode_grid_local.json",
    gpu_type = "H20",
    gpu_count = 2,
)

py_test(
    name = "decode_remote_test_test",
    srcs = ["decode_remote_test_test.py", "decode_remote_test.py"],
    data = ["aone_defs.bzl"],
    deps = [":perf_test_support", ":perf_test_runtime_lib"],
)

py_test(
    name = "aone_entry_test",
    srcs = ["aone_entry_test.py"],
    deps = [":aone_lib"],
)

py_test(
    name = "policy_extraction_test",
    srcs = ["policy_extraction_test.py"],
    deps = [":perf_test_regression_lib", ":perf_test_publish_policy_lib"],
)

py_test(
    name = "regression_test",
    srcs = ["regression_test.py"],
    deps = [":perf_test_regression_lib", ":perf_test_regression_test_support"],
)

py_test(
    name = "publication_policy_test",
    srcs = ["publication_policy_test.py"],
    deps = [":perf_test_publish_policy_lib"],
)
