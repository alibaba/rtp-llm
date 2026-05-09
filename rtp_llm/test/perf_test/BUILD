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
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "1",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "DSV4_USE_MEGA_MOE": "1",
    },
    tags = ["manual"],
)

# V4-Flash CP=4 + EP=4 + Mega MoE prefill profile.  Same topology as
# the baseline above but with DSV4_USE_MEGA_MOE=1 + DG_JIT_CPP_STANDARD=20.
# Smaller input (8k vs baseline 64k) because Mega MoE's symm-mem buffer
# allocation is sized from max_tokens_per_rank=max_seq_len; at 64k the
# symm allocation aborts in CUDASymmetricMemory destructor during warmup.
# 8k still exercises the Mega MoE fused dispatch + L1/L2 GEMM + combine
# path on every MoE layer — enough kernel data to compare against the
# baseline's per-layer DeepEP dispatch+combine breakdown.
py_test(
    name = "v4_flash_cp4_ep4_prefill_8k_mega_perf",
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
        "--input_len", "8192",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "64",
        "--tp_size", "4",
        "--dp_size", "1",
        "--ep_size", "4",
        "--world_size", "4",
        "--max_seq_len", "8200",
        "--cp_rotate_method", "ALL_GATHER",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "1",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "DSV4_USE_MEGA_MOE": "1",
    },
    tags = ["manual"],
)

# V4-Flash DP=4 + EP=4 captured-graph decode grid sweep.  Mirrors the
# topology of smoke target v4_flash_decode_dp4_ep4_graph_mega_concurrent_sm100
# (TP=1 / DP=4 / EP=4 / world=4, DeepEP non-low-latency, Mega MoE via
# CUDA graph capture at BS=1,2,4,8) and drives the perf framework over
# the same batch_size grid so per-BS decode timelines land in
# TEST_UNDECLARED_OUTPUTS_DIR.  Local checkpoint avoids ModelScope.
py_test(
    name = "v4_flash_dp4_decode_grid_perf",
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
        "--input_len", "4096,8192",
        "--partial", "1",
        "--decode_test_length", "8",
        "--seq_size_per_block", "64",
        "--max_seq_len", "16384",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1"
    ],
    env = {
        "WORLD_SIZE": "4",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "DSV4_USE_MEGA_MOE": "1",
    },
    tags = ["manual"],
)

# DeepSeek V4-Flash perf coverage on SM100 (B300 / GB200).
#   v4_flash_cp4_ep4_prefill_64k_perf  → CP=4 EP=4 64k prefill timeline
#   v4_flash_dp4_decode_grid_perf      → DP=4 EP=4 captured-graph decode grid
test_suite(
    name = "perf_dsv4_v4_flash",
    tests = [
        ":v4_flash_cp4_ep4_prefill_64k_perf",
        ":v4_flash_dp4_decode_grid_perf",
    ],
    tags = ["manual"],
)
