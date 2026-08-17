"""Compact SM120 DeepSeek-V4-Flash standalone perf targets."""

load("@rules_python//python:defs.bzl", "py_test")

_COMMON_DEPS = [
    "//rtp_llm:pyodps",
    "//rtp_llm:testlib",
    "//rtp_llm/test/perf_test:perf_test_lib",
]

def sm120_dsv4_perf(name, topology, phase, speculative = True):
    """Defines one standalone SM120 prefill or decode perf/profile target."""
    if topology not in ("cp4_dp4", "cp8_dp8", "tp4", "tp8"):
        fail("unsupported topology: " + topology)
    if phase not in ("prefill", "decode"):
        fail("unsupported phase: " + phase)

    prefill = phase == "prefill"
    cp_dp = topology in ("cp4_dp4", "cp8_dp8")
    degree = 8 if topology in ("cp8_dp8", "tp8") else 4
    tp_size = degree if prefill or not cp_dp else 1
    dp_size = degree if not prefill and cp_dp else 1

    args = [
        "--model_type", "deepseek_v4",
        "--checkpoint_path", "/home/tanboyu.tby/models/DeepSeek-V4-Flash-0731",
        "--tokenizer_path", "/home/tanboyu.tby/models/DeepSeek-V4-Flash-0731",
        "--batch_size", "1",
        "--input_len", "8192,32768",
        "--partial", "2" if prefill else "1",
        "--decode_test_length", "2" if prefill else "64",
        "--max_seq_len", "32832",
        "--seq_size_per_block", "256",
        "--kernel_seq_size_per_block", "128",
        "--tp_size", str(tp_size),
        "--dp_size", str(dp_size),
        "--ep_size", str(degree),
        "--world_size", str(degree),
        "--use_deepep_moe", "1",
        "--use_deepep_low_latency", "0" if prefill else "1",
        "--act_type", "BF16",
        "--fp8_kv_cache", "1",
        "--load_method", "fastsafetensors",
        "--concurrency_limit", "1" if topology == "tp4" else ("1" if prefill else "4"),
        "--max_context_batch_size", "1",
        "--frontend_server_count", "1",
        "--enable_cuda_graph", "0" if prefill or (speculative and cp_dp) else "1",
        "--decode_capture_config",
        "1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96",
        "--reserver_runtime_mem_mb", "8192",
    ]
    if prefill and cp_dp:
        args += ["--cp_rotate_method", "ALL_GATHER"]
    if not prefill and speculative:
        args += [
            "--sp_type", "dspark",
            "--gen_num_per_cycle", "3",
            "--sp_model_type", "deepseek_v4_dspark",
            "--sp_checkpoint_path", "/home/tanboyu.tby/models/DeepSeek-V4-Flash-0731",
            "--sp_act_type", "bf16",
        ]

    env = {
        "WORLD_SIZE": str(degree),
        "DSV4_BF16_VLLM": "0",
        "DSV4_FIXED_POOL_BLOCKS": "256" if cp_dp else "128",
        "DSV4_FUSED_PREPARE": "1",
        "DSV4_MOE_CHUNK_TOKENS": "4096",
        "DSV4_SM120_NCCL_EP_PIPELINE": "1",
        "ENABLE_FP32_LM_HEAD": "0",
        "GEN_TIMELINE_SYNC": "1",
        "PERF_GRID_WARMUP_RUNS": "1",
        "PERF_FORMAL_WARMUP_RUNS": "1",
        "PERF_MEASURE_RUNS": "3",
        "PERF_PREARM_PROFILE": "1",
        "PERF_PROFILE_ARM_SLEEP": "2",
        "PERF_PROFILE_FLUSH_SLEEP": "15",
        "PERF_PROFILE_NUM_STEPS": "1" if prefill else "6",
        "PERF_PROFILE_RUNS": "1",
        "PERF_RANDOM_SEED": "20260817",
    }
    if prefill:
        env["DSV4_CHUNK_TOKENS"] = "12288" if cp_dp else "4096"
    else:
        env["FLASHMLA_FORCE_HEAD64X2"] = "1"
    if prefill and cp_dp:
        env["DSV4_PREFILL_CP_OVERLAP"] = "1"
        env["PREFILL_CP_KV_CACHE_SHARDED"] = "1"
    # These targets exercise one PCIe-only RTX PRO 5000 host. GPU peer access
    # is substantially faster than host-staged copies on this topology; only
    # disable inter-node IB discovery.
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_MIN_P2P_NCHANNELS"] = "8"
    env["NCCL_MAX_P2P_NCHANNELS"] = "8"

    py_test(
        name = name,
        main = "batch_decode_test.py",
        srcs = ["batch_decode_test.py"],
        timeout = "eternal",
        deps = _COMMON_DEPS,
        data = ["//rtp_llm:sdk"],
        args = args,
        env = env,
        tags = ["manual", "sm120"],
    )
