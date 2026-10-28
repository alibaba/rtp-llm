package(default_visibility = ["//visibility:public"])
load("//rtp_llm/test/perf_test:defs.bzl", "perf_test")

# Exported so the perf_test() macro (defined in defs.bzl, instantiated by
# internal_source/rtp_llm/test/perf_test/BUILD) can reference test_entry.py
# across packages.
exports_files([
    "test_entry.py",
    "batch_decode_test.py",
])

py_library(
    name = "perf_test_lib",
    srcs = [
        "batch_decode_test.py",
        "batch_perf_impl.py",
        "cache_grid_runner.py",
        "dataclass.py",
        "dataset.py",
        "deepseek_v4_prefill_formula_fit.py",
        "distribution_runner.py",
        "generate_prefill_3d_chart.py",
        "grid_runner.py",
        "hub_download.py",
        "offline_bench_test.py",
        "offline_runner.py",
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
    name = "batch_decode_test_test",
    srcs = ["batch_decode_test_test.py"],
    deps = [
        ":perf_test_lib",
    ],
)

py_test(
    name = "test_perf_config_and_runners",
    srcs = ["test_perf_config_and_runners.py"],
    data = [
        "//rtp_llm/test/tokenizer_test:testdata",
    ],
    deps = [
        "//rtp_llm/test/perf_test:perf_test_lib",
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
        "//rtp_llm:pyodps",
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

# Prefix-cache mode: measures an explicit total-sequence × cache-length grid.
# Invoke with --test_arg=--cache_grid_json=<file> and --test_arg=--partial=2.
# The runner seeds each requested prefix and verifies reuse_len on every
# measured request; cache_len is a workload dimension, not an engine argument.
py_test(
    name = "cache_grid_perf_test",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    tags = ["manual"],
)

# Generic model-agnostic entrypoint. It intentionally has no model-specific
# defaults; pass model/workload/engine settings with --test_arg at invocation.
py_test(
    name = "model_perf_test",
    main = "batch_decode_test.py",
    srcs = ["batch_decode_test.py"],
    timeout = "eternal",
    deps = [
        "//rtp_llm:pyodps",
        "//rtp_llm:testlib",
        "//rtp_llm/test/perf_test:perf_test_lib",
    ],
    data = ["//rtp_llm:sdk"],
    tags = ["manual"],
)

# Public PPU Qwen3.5-MoE performance coverage. These targets are manual and
# require the corresponding model weights on a PPU worker.
perf_test(
    name = "qwen35_moe_ppu_decode_dp8_ll_cg_full_grid",
    model_type = "qwen35_moe",
    ckpt_path = "/mnt/nas1/hf/Qwen3.5-35B-A3B",
    gpu_type = "PPU-ZW810E",
    envs = {"LOAD_METHOD": "scratch"},
    args = [
        "--input_len", "1024,4096,8192,16384,32768,65536,131072",
        "--batch_size", "1,8,16,32,64,128",
        "--partial", "1", "--concurrency_limit", "128",
        "--decode_test_length", "30", "--num_measures", "5",
        "--act_type", "BF16", "--seq_size_per_block", "2048",
        "--kernel_seq_size_per_block", "128",
        "--reserver_runtime_mem_mb", "4096", "--tp_size", "1",
        "--dp_size", "8", "--world_size", "8",
        "--use_deepep_moe", "1", "--use_deepep_low_latency", "1",
        "--enable_cuda_graph", "1",
        "--decode_capture_config", "1,8,16,32,64,128",
    ],
)

perf_test(
    name = "qwen35_moe_ppu_decode_dp8_seq65k",
    model_type = "qwen35_moe",
    ckpt_path = "/mnt/nas1/hf/Qwen3.5-35B-A3B",
    gpu_type = "PPU-ZW810E",
    envs = {"LOAD_METHOD": "scratch"},
    args = [
        "--input_len", "65536", "--batch_size", "1,2,4,8,16,32,64,128",
        "--partial", "1", "--concurrency_limit", "128",
        "--decode_test_length", "30", "--num_measures", "5",
        "--act_type", "BF16", "--seq_size_per_block", "2048",
        "--kernel_seq_size_per_block", "128",
        "--reserver_runtime_mem_mb", "4096", "--tp_size", "1",
        "--dp_size", "8", "--world_size", "8",
        "--use_deepep_moe", "1", "--use_deepep_low_latency", "1",
        "--enable_cuda_graph", "1",
        "--decode_capture_config", "1,2,4,8,16,32,64,128",
    ],
)

perf_test(
    name = "qwen35_moe_ppu_decode_dp8_seq32k",
    model_type = "qwen35_moe",
    ckpt_path = "/mnt/nas1/hf/Qwen3.5-35B-A3B",
    gpu_type = "PPU-ZW810E",
    envs = {"LOAD_METHOD": "scratch"},
    args = [
        "--input_len", "32768", "--batch_size", "1,2,4,8,16,32,64,128",
        "--partial", "1", "--concurrency_limit", "128",
        "--decode_test_length", "30", "--num_measures", "5",
        "--act_type", "BF16", "--seq_size_per_block", "2048",
        "--kernel_seq_size_per_block", "128",
        "--reserver_runtime_mem_mb", "4096", "--tp_size", "1",
        "--dp_size", "8", "--world_size", "8",
        "--use_deepep_moe", "1", "--use_deepep_low_latency", "1",
        "--enable_cuda_graph", "1",
        "--decode_capture_config", "1,2,4,8,16,32,64,128",
    ],
)

perf_test(
    name = "qwen35_moe_ppu_decode_dp8_ll_cg_kspb128",
    model_type = "qwen35_moe",
    ckpt_path = "/mnt/nas1/hf/Qwen3.5-35B-A3B",
    gpu_type = "PPU-ZW810E",
    envs = {"LOAD_METHOD": "scratch"},
    args = [
        "--input_len", "8192", "--batch_size", "1,128",
        "--partial", "1", "--concurrency_limit", "128",
        "--decode_test_length", "30", "--num_measures", "5",
        "--act_type", "BF16", "--seq_size_per_block", "2048",
        "--kernel_seq_size_per_block", "128",
        "--reserver_runtime_mem_mb", "4096", "--tp_size", "1",
        "--dp_size", "8", "--world_size", "8",
        "--use_deepep_moe", "1", "--use_deepep_low_latency", "1",
        "--enable_cuda_graph", "1",
        "--decode_capture_config", "1,8,16,32,64,128",
    ],
)

perf_test(
    name = "qwen35_moe_ppu_prefill_tp8_deepep_normal_deepgemm_contiguous_v3",
    model_type = "qwen35_moe",
    ckpt_path = "/mnt/nas1/hf/Qwen3.5-35B-A3B",
    gpu_type = "PPU-ZW810E",
    envs = {"LOAD_METHOD": "scratch"},
    args = [
        "--input_len", "1024,2048,4096,8192,16384,32768,65536",
        "--partial", "2", "--concurrency_limit", "8",
        "--num_measures", "5", "--warm_up", "0",
        "--act_type", "BF16", "--seq_size_per_block", "2048",
        "--kernel_seq_size_per_block", "128",
        "--reserver_runtime_mem_mb", "4096", "--tp_size", "8",
        "--world_size", "8", "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--moe_strategy", "no_quant_dp_normal_deepgemm",
    ],
)

test_suite(
    name = "perf_ppu_qwen35",
    tests = [
        ":qwen35_moe_ppu_decode_dp8_ll_cg_full_grid",
        ":qwen35_moe_ppu_decode_dp8_seq65k",
        ":qwen35_moe_ppu_decode_dp8_seq32k",
        ":qwen35_moe_ppu_decode_dp8_ll_cg_kspb128",
        ":qwen35_moe_ppu_prefill_tp8_deepep_normal_deepgemm_contiguous_v3",
    ],
    tags = ["manual"],
)
