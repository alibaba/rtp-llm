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

# V4-Flash CP=4 + EP=4 prefill profile on 200k input.  Same topology as
# v4_flash_cp4_ep4_prefill_64k_perf, with a longer single query for
# long-context chunk-size experiments.
py_test(
    name = "v4_flash_cp4_ep4_prefill_200k_perf",
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
        "--input_len", "200000",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "64",
        "--tp_size", "4",
        "--dp_size", "1",
        "--ep_size", "4",
        "--world_size", "4",
        "--max_seq_len", "200064",
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

# V4-Flash DP=4 + EP=4 captured-graph decode at fixed bs=128 / input_len=64K.
# 64K context x 128 batch (per DP rank) x DP=4 x FP8 KV x FP8_PER_BLOCK weights
# x Mega MoE x cuda graph capture. Canonical iter target for feat/dsv4_on_dev
# decode perf-tuning (see dpskv4-study/perf/dsv4_on_dev_64k/iterations.md).
# --batch_size is per-DP-rank: grid_runner.py:84 multiplies by --dp_size before
# handing off to BatchPerfImpl, which then divides back by dp_size on the
# engine side (batch_perf_impl.py:182).  Net: --batch_size 128 + --dp_size 4
# = 512 total requests distributed 128/rank across 4 DP ranks.  server.py:63
# also overrides --concurrency_limit to max(batch_size_list)=128 at engine
# startup, so the BUILD --concurrency_limit below is effectively ignored.
# State/indexer pools sized by CacheConfig.h:114 as
# 2 * max(max_context_batch_size, max_generate_batch_size) + 1.  With
# concurrency_limit=128 driving max_generate_batch_size=128 and the default
# max_context_batch_size=1, the fixed pools end up exactly 257 blocks ==
# 128 reqs * 2 blocks + 1.  Block 0 is reserved by BlockPool, so usable is 256
# == zero margin; any 5% reserve_block_ratio haircut (which --reserve_block_ratio 0
# does not fully disable — the default still propagates through some config copies)
# drops admission to ~85/128 per rank (340/512 total, same 66% failure as
# baseline).  Bumping --max_context_batch_size to 256 lifts the state-pool
# ceiling to 2*256+1=513 blocks so 128 concurrent reqs fit with ~2x headroom,
# and the fixed-pool memory overhead is ~9 MiB per rank (negligible vs the
# ~169 GiB paged KV pools).
py_test(
    name = "v4_flash_dp4_decode_bs128_64k_perf",
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
        "--checkpoint_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--tokenizer_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--batch_size", "128",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "64",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--reserve_block_ratio", "0",
        "--max_context_batch_size", "256",
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
        "ENABLE_FP32_LM_HEAD": "0",
        "DSV4_FUSED_PREPARE": "1",
    },
    tags = ["manual"],
)

# No-graph sibling for diagnosing cuda-graph capture failures.
py_test(
    name = "v4_flash_dp4_decode_bs128_64k_perf_nograph",
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
        "--checkpoint_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--tokenizer_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--batch_size", "128",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "64",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "0",
        "--reserver_runtime_mem_mb", "32768",
        "--reserve_block_ratio", "0",
        "--max_context_batch_size", "256",
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
        "DSV4_FUSED_PREPARE": "1",
    },
    tags = ["manual"],
)

# Diagnostic probe: bs=16 / 64K with cuda graph off — smallest bs that still
# uses 4-rank DP/EP topology, to confirm whether decode warmup regression is
# bs-threshold specific (bs=128/64 both fail on feat/dsv4_on_dev HEAD).
py_test(
    name = "v4_flash_dp4_decode_bs16_64k_perf_probe",
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
        "--checkpoint_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--tokenizer_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--batch_size", "16",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "64",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
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
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_USE_CUDA_DSA": "1",
    },
    tags = ["manual"],
)

py_test(
    name = "v4_flash_dp4_decode_bs64_64k_perf_probe",
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
        "--checkpoint_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--tokenizer_path", "/home/baowending.bwd/DeepSeek-V4-Flash",
        "--batch_size", "64",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "64",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
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

# V4-Flash DP=4 + EP=4 captured-graph decode grid sweep with MTP
# speculative decoding + FP8 KV cache.  Mirrors the decode side of the
# smoke target ``v4_flash_pd_cp2ep2_dp2ep2_sm100_fp8kv1_mtp_batch4_cudagraph``
# (TP=1 / DP=4 / EP=4 / world=4, DeepEP low-latency, decode CUDA graph,
# MTP draft via deepseek_v4_mtp + gen_num_per_cycle=1, FP8 KV) but
# scaled out from DP=2 → DP=4 and pinned at input_len=65536 (64K).
# Sweeps ``--batch_size`` over 1, 8, 16, 32 so per-BS MTP+graph decode
# timelines land in TEST_UNDECLARED_OUTPUTS_DIR.  ``decode_capture_config``
# captures one graph per BS (matches the grid).
py_test(
    name = "v4_flash_dp4_decode_64k_mtp_grid_perf",
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
        "--batch_size", "1,8,16,32",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "30",
        "--seq_size_per_block", "64",
        "--tp_size", "1",
        "--dp_size", "4",
        "--ep_size", "4",
        "--world_size", "4",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
        "--sp_type", "mtp",
        "--gen_num_per_cycle", "3",
        "--sp_model_type", "deepseek_v4_mtp",
        "--sp_checkpoint_path", "/mnt/nas1/hf/DeepSeek-V4-Flash",
        "--sp_act_type", "bf16",
        "--warm_up", "0",
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

# DeepSeek V4-Pro CP=8 + EP=8 prefill profile on 64k input.
py_test(
    name = "v4_pro_cp8_ep8_prefill_64k_perf",
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
        "--checkpoint_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--tokenizer_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--batch_size", "1",
        "--input_len", "65536",
        "--partial", "2",
        "--decode_test_length", "2",
        "--seq_size_per_block", "64",
        "--tp_size", "8",
        "--dp_size", "1",
        "--ep_size", "8",
        "--world_size", "8",
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
        "WORLD_SIZE": "8",
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

# DeepSeek V4-Pro DP=8 + EP=8 captured-graph decode grid sweep.
py_test(
    name = "v4_pro_dp8_decode_grid_perf",
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
        "--checkpoint_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--tokenizer_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--batch_size", "1,32,64,128,256",
        "--input_len", "4096,8192",
        "--partial", "1",
        "--decode_test_length", "8",
        "--seq_size_per_block", "64",
        "--max_seq_len", "16384",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--fp8_kv_cache", "1",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--max_context_batch_size", "1",
    ],
    env = {
        "WORLD_SIZE": "8",
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

# DeepSeek V4-Pro DP=8 + EP=8 captured-graph decode at fixed bs=128 / input_len=64K.
py_test(
    name = "v4_pro_dp8_decode_bs128_64k_perf",
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
        "--checkpoint_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--tokenizer_path", "/data1/serina.wzq/DeepSeek-V4-Pro",
        "--batch_size", "128",
        "--input_len", "65536",
        "--partial", "1",
        "--decode_test_length", "16",
        "--seq_size_per_block", "64",
        "--max_seq_len", "65664",
        "--tp_size", "1",
        "--dp_size", "8",
        "--ep_size", "8",
        "--world_size", "8",
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0",
        "--act_type", "BF16",
        "--quantization", "FP8_PER_BLOCK",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "32",
        "--enable_cuda_graph", "1",
        "--reserver_runtime_mem_mb", "32768",
        "--reserve_block_ratio", "0",
        "--max_context_batch_size", "256",
    ],
    env = {
        "WORLD_SIZE": "8",
        "DG_JIT_CPP_STANDARD": "20",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_NUM_STEPS": "4",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "120",
        "DSV4_USE_MEGA_MOE": "1",
        "ENABLE_FP32_LM_HEAD": "0",
        "DSV4_FUSED_PREPARE": "1",
    },
    tags = ["manual"],
)

test_suite(
    name = "perf_dsv4_v4_pro",
    tests = [
        ":v4_pro_cp8_ep8_prefill_64k_perf",
        ":v4_pro_dp8_decode_grid_perf",
        ":v4_pro_dp8_decode_bs128_64k_perf",
    ],
    tags = ["manual"],
)

# DeepSeek V4-Flash perf coverage on SM100 (B300 / GB200).
#   v4_flash_cp4_ep4_prefill_64k_perf       → CP=4 EP=4 64k prefill timeline
#   v4_flash_cp4_ep4_prefill_200k_perf      → CP=4 EP=4 200k prefill timeline
#   v4_flash_dp4_decode_grid_perf           → DP=4 EP=4 captured-graph decode grid
#   v4_flash_dp4_decode_bs128_64k_perf      → DP=4 EP=4 fixed bs=128/input=64K (perf-tuning canonical)
#   v4_flash_dp4_decode_64k_mtp_grid_perf   → DP=4 EP=4 + MTP + cudagraph @ 64K, BS={1,8,16,32}
test_suite(
    name = "perf_dsv4_v4_flash",
    tests = [
        ":v4_flash_cp4_ep4_prefill_64k_perf",
        ":v4_flash_cp4_ep4_prefill_200k_perf",
        ":v4_flash_dp4_decode_grid_perf",
        ":v4_flash_dp4_decode_bs128_64k_perf",
        ":v4_flash_dp4_decode_64k_mtp_grid_perf",
    ],
    tags = ["manual"],
)
