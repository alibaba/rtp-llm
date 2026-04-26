package(default_visibility = ["//visibility:public"])

# Exported so the perf_test() macro (defined in defs.bzl, instantiated by
# internal_source/rtp_llm/test/perf_test/BUILD) can reference test_entry.py
# across packages.
exports_files([
    "test_entry.py",
])

py_library(
    name = "perf_test_lib",
    srcs = [
        "batch_decode_test.py",
        "batch_perf_impl.py",
        "dataclass.py",
        "dataset.py",
        "distribution_runner.py",
        "grid_runner.py",
        "hub_download.py",
        "perf_config.py",
        "perf_utils.py",
        "tps_runner.py",
        "sampling.py",
        "server.py",
        "test_util.py",
        "visualization.py",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:device_resource",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_perf_config_and_runners",
    srcs = ["test_perf_config_and_runners.py"],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
)

# Distribution mode: sampling from dataset using model's parameters.
py_test(
        name = "distribution_perf_test",
        main = "batch_decode_test.py",
        srcs = [
            "batch_decode_test.py",
        ],
        timeout = "eternal",
        deps = [
            "//rtp_llm:pyodps",
            "//rtp_llm:testlib",
            "//rtp_llm/test/perf_test:perf_test_lib",
        ],
        data = [
            "//rtp_llm:sdk",
        ],
        args = [
            "--checkpoint_path", "Qwen/Qwen3.5-35B-A3B",
            "--dataset_name", "sharegpt",
            "--max_seq_len", "8192",
            "--concurrency_limit", "64",
            "--seq_size_per_block", "1024"
        ],
        tags = ["manual"],
    )

# Grid mode: perf test with fixed batch_size × input_len.
py_test(
    name = "grid_perf_test",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type",
        "qwen35_moe",
        "--checkpoint_path",
        "Qwen/Qwen3.5-35B-A3B",
        "--batch_size",
        "1",
        "--input_len",
        "128",
        "--partial",
        "1",
        "--decode_test_length",
        "8",
        "--seq_size_per_block",
        "1024",
        "--tp_size",
        "1",
        "--dp_size",
        "1",
    ],
    tags = ["manual"],
)

# V4-Flash CP=4 + EP=4 prefill profile on 64k input.  Manual-only —
# drives a single batch_size=1, input_len=65536 prefill through the
# perf test framework so the timeline JSONs land in TEST_UNDECLARED_
# OUTPUTS_DIR.  Uses a local checkpoint path so the perf hub resolver
# does not try ModelScope.
py_test(
    name = "v4_flash_cp4_ep4_prefill_64k_perf",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--tokenizer_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--batch_size", "1",
        "--input_len", "65536",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "64",
        "--tp_size", "4",
        "--dp_size", "1",
        "--ep_size", "4",
        "--world_size", "4",
        "--max_seq_len", "65600",
        "--cp_rotate_method", "ALL_GATHER",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--load_method", "scratch",
        "--concurrency_limit", "1",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    tags = ["manual"],
)
