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
        "sampling.py",
        "server.py",
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

# Grid mode:固定 batch_size × input_len 扫面，不传 dataset / test_json。
# 需显式 --model_type；远端 checkpoint 由 batch_decode_test 在启动引擎前解析为本地路径。
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
