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
        "case_planning.py",
        "cli.py",
        "config_resolver.py",
        "config_schema.py",
        "decode/__init__.py",
        "decode/metrics.py",
        "decode/model.py",
        "decode/planner.py",
        "decode/scheduler.py",
        "entry.py",
        "execution/local_bazel.py",
        "errors.py",
        "experiment.py",
        "launch_config.py",
        "plan.py",
        "profiling.py",
        "prefill/__init__.py",
        "prefill/block_size.py",
        "prefill/export.py",
        "prefill/fingerprints.py",
        "prefill/measurement.py",
        "prefill/logarithmic_grid.py",
        "prefill/planner.py",
        "prefill/prompts.py",
        "prefill/random_batch.py",
        "prefill/traffic_batch.py",
        "prefill/traffic_distribution.py",
        "run_context.py",
        "run_directory.py",
        "ssh/__init__.py",
        "ssh/config_schema.py",
        "ssh/plan.py",
        "ssh/source.py",
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
        "decode/transport.py",
        "execution/artifacts.py",
        "execution/devices.py",
        "execution/distributed/__init__.py",
        "execution/distributed/decode.py",
        "execution/distributed/decode_sync.py",
        "execution/distributed/member.py",
        "execution/distributed/prefill.py",
        "execution/distributed/rank_group.py",
        "execution/engine_session.py",
        "execution/gang.py",
        "execution/launch_boundary.py",
        "execution/local.py",
        "execution/runtime.py",
        "modes.py",
        "prefill/chart.py",
        "prefill/checkpoint.py",
        "prefill/dashsc_transport.py",
        "prefill/finalize.py",
        "prefill/formula.py",
        "prefill/probe.py",
        "prefill/profile.py",
        "prefill/report.py",
        "prefill/runner.py",
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
    "BUILD",
    "aone/gang_worker.py",
    "aone/worker.py",
    "aone_defs.bzl",
    "decode_remote_test.py",
    "examples/decode_aone_gang.json",
    "examples/decode_distribution_local.json",
    "examples/decode_distribution_requests.json",
    "examples/decode_grid_local.json",
    "examples/prefill_aone.json",
    "examples/prefill_aone_ci.json",
    "examples/prefill_local.json",
    "examples/prefill_multi_grid_local.json",
    "examples/qwen35_prefill_test.json",
    "examples/req_distribution.json",
    "main.py",
    "local_main.py",
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
        "engine_paths.py",
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

py_binary(
    name = "perf_test_golden",
    main = "regression/golden.py",
    srcs = ["regression/golden.py"],
    deps = [":perf_test_regression_lib"],
)

# Aggregate for standalone offline callers and CI helpers.
py_library(
    name = "perf_test_ssh_lib",
    srcs = [
        "execution/ssh.py",
        "ssh/remote.py",
        "ssh/worker.py",
    ],
    deps = [":perf_test_entry_lib", ":perf_test_runtime_lib", ":perf_test_results_lib"],
)

py_library(
    name = "perf_test_lib",
    srcs = [
        "offline_bench_test.py",
        "offline_runner.py",
        "server.py",
    ],
    data = [":perf_test_examples"],
    deps = [
        ":perf_test_ssh_lib",
        ":aone_lib",
        ":perf_test_entry_lib",
        ":perf_test_runtime_lib",
        ":perf_test_results_lib",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:device_resource",
    ],
)

# Site-CI adapters: deliver pipeline parameters onto a strict config and
# gate collected prefill artifacts. Portable helpers; CI invokes them
# explicitly, they never run as part of a normal perf test.
py_library(
    name = "ci_helpers",
    srcs = [
        "ci_delivery.py",
        "ci_prefill_verify.py",
    ],
    data = [
        "examples/decode_aone_gang.json",
        "examples/prefill_aone_ci.json",
    ],
    deps = [
        ":perf_test_lib",
    ],
)

py_binary(
    name = "ci_delivery",
    srcs = ["ci_delivery.py"],
    deps = [":ci_helpers"],
)

py_binary(
    name = "ci_prefill_verify",
    srcs = ["ci_prefill_verify.py"],
    deps = [":ci_helpers"],
)

# Internal worker: tools/perf_test passes a frozen --run-bundle as a test_arg.
# Reuse .bazelrc test_env and run_under instead of duplicating toolchain setup.
py_test(
    name = "perf_test_local",
    srcs = ["local_main.py"],
    main = "local_main.py",
    timeout = "eternal",
    tags = ["manual", "local"],
    data = [
        "//rtp_llm:sdk",
        ":perf_test_examples",
    ],
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
    name = "export_prefill_batches",
    main = "prefill/export.py",
    srcs = ["prefill/export.py"],
    deps = [":perf_test_entry_lib"],
)

# One test action runs the complete local-backend grid on a worker with two GPUs.
# Adapt the example model/tokenizer paths to files readable on that worker.
unified_decode_remote_test(
    name = "decode_remote_example",
    config = "examples/decode_grid_local.json",
    gpu_type = "H20_CU13",
    gpu_count = 2,
)

filegroup(
    name = "perf_test_python_sources",
    srcs = glob(["**/*.py"], exclude = ["tests/**"]),
)

py_library(
    name = "perf_test_analyzer_lib",
    srcs = [
        "multi_node/analyzer/batch_analyze.py",
        "multi_node/analyzer/configs/module_base.py",
        "multi_node/analyzer/configs/module_conf.py",
        "multi_node/analyzer/configs/qwen_480b_decode-fp8.py",
        "multi_node/analyzer/configs/qwen_480b_prefill_fp8.py",
        "multi_node/analyzer/trace_analyze.py",
    ],
    deps = [
        ":perf_test_entry_lib",
    ],
)

test_suite(
    name = "unit_tests",
    tests = ["//rtp_llm/test/perf_test/tests:unit_tests"],
)

filegroup(
    name = "perf_test_examples",
    srcs = [
        "examples/decode_aone_gang.json",
        "examples/decode_distribution_local.json",
        "examples/decode_distribution_requests.json",
        "examples/decode_grid_local.json",
        "examples/decode_ssh_experiment.json",
        "examples/decode_ssh_gang.json",
        "examples/prefill_aone.json",
        "examples/prefill_aone_ci.json",
        "examples/prefill_local.json",
        "examples/prefill_shared.json",
        "examples/prefill_multi_grid_local.json",
        "examples/qwen35_prefill_test.json",
        "examples/req_distribution.json",
    ],
)

load("//rtp_llm/test/perf_test:suites_cu13_glm5x.bzl", "cu13_glm5x_perf_suites")

cu13_glm5x_perf_suites()

# Private upstream GLM grid compatibility; no generic legacy executable target.
py_library(
    name = "glm5x_legacy_lib",
    srcs = [
        "glm5x_legacy/__init__.py",
        "glm5x_legacy/batch_decode_test.py",
        "glm5x_legacy/batch_perf_impl.py",
        "glm5x_legacy/dataclass.py",
        "glm5x_legacy/grid_runner.py",
        "glm5x_legacy/perf_config.py",
        "glm5x_legacy/perf_utils.py",
        "glm5x_legacy/test_entry.py",
        "glm5x_legacy/tps_runner.py",
        "glm5x_legacy/visualization.py",
    ],
    deps = [":perf_test_lib"],
    visibility = ["//visibility:private"],
)

# GLM5.x policy is opt-in through its dedicated entry; other perf suites do not use it.
py_library(
    name = "glm5x_perf_lib",
    srcs = ["glm5x_perf.py"],
    deps = [":glm5x_legacy_lib"],
)

py_test(
    name = "glm5x_perf_test",
    srcs = ["glm5x_perf_test.py"],
    deps = [":glm5x_perf_lib", ":glm5x_legacy_lib"],
)
