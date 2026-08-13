"""SM120 DeepSeek-V4-Flash prefill-only perf target macro."""

load("@rules_python//python:defs.bzl", "py_test")

def sm120_dsv4_prefill_perf(name, parallel_size, use_cp):
    engine_args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/home/tanboyu.tby/models/DeepSeek-V4-Flash-0731",
        "--tokenizer_path", "/home/tanboyu.tby/models/DeepSeek-V4-Flash-0731",
        "--batch_size", "1", "--input_len", "65536", "--partial", "2",
        "--decode_test_length", "2", "--seq_size_per_block", "256",
        "--kernel_seq_size_per_block", "128", "--tp_size", str(parallel_size),
        "--dp_size", "1", "--ep_size", str(parallel_size),
        "--world_size", str(parallel_size), "--max_seq_len", "65538",
        "--use_deepep_moe", "1", "--use_deepep_low_latency", "0",
        "--act_type", "BF16", "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors", "--concurrency_limit", "1",
        "--enable_cuda_graph", "0", "--reserver_runtime_mem_mb", "8192",
        "--max_context_batch_size", "1", "--frontend_server_count", "1",
    ]
    if use_cp:
        engine_args += ["--cp_rotate_method", "ALL_GATHER"]
    target_env = {
        "WORLD_SIZE": str(parallel_size), "DSV4_FIXED_POOL_BLOCKS": "256",
        "DSV4_MOE_CHUNK_TOKENS": "4096", "DSV4_SM120_FLASHINFER_PAGE64": "1",
        "DSV4_BF16_VLLM": "0", "DSV4_FUSED_PREPARE": "1",
        "ENABLE_FP32_LM_HEAD": "0", "PERF_GRID_WARMUP_RUNS": "1",
        "PERF_FORMAL_WARMUP_RUNS": "1", "PERF_MEASURE_RUNS": "3",
        "PERF_PROFILE_RUNS": "1", "PERF_PROFILE_FLUSH_SLEEP": "120",
        "GEN_TIMELINE_SYNC": "1",
    }
    if use_cp:
        target_env["PREFILL_CP_KV_CACHE_SHARDED"] = "1"
    py_test(
        name = name, main = "batch_decode_test.py", srcs = ["batch_decode_test.py"],
        timeout = "eternal",
        deps = ["//rtp_llm:pyodps", "//rtp_llm:testlib", "//rtp_llm/test/perf_test:perf_test_lib"],
        data = ["//rtp_llm:sdk"], args = engine_args, env = target_env,
        tags = ["manual", "sm120"],
    )
