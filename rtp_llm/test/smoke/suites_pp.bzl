"""PP smoke with optional chunked prefill on the same representative topologies."""

load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

_COMMON_ARGS = "--act_type BF16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10"
# Qwen3.5 recurrent state uses 2048-token blocks; attention uses 64-token pages.
_HYBRID_ARGS = " --model_warm_up 0 --ssm_state_dtype fp32 --kv_cache_mem_mb 8192 --kernel_seq_size_per_block 64"

def _pp_test(name, gpus, cases, gpu_type, hybrid = False, extra_args = "", chunk_size = 0):
    chunk_args = ""
    if chunk_size:
        chunk_args = " --warm_up 1 --max_context_batch_size 8 --max_batch_tokens_size 65536 --prefill_chunk_size %d" % chunk_size
    custom_smoke_test(
        name = name,
        main = "pp_test.py",
        args = ["--cases=" + ",".join(cases)],
        smoke_args = "%s --world_size %d --max_seq_len %d --seq_size_per_block %d%s%s%s" % (
            _COMMON_ARGS,
            gpus,
            chunk_size * 4 if chunk_size else 4096,
            2048 if hybrid else 16,
            _HYBRID_ARGS if hybrid else "",
            extra_args,
            chunk_args,
        ),
        gpu_type = gpu_type,
    )

def pp_suites(gpu_type):
    _pp_test(
        "qwen35_pp_depth_test",
        4,
        ["pdfusion_pp4_tp1"],
        gpu_type,
        hybrid = True,
    )
    _pp_test(
        "qwen35_pp_mixed_mtp_test",
        8,
        ["pdfusion_pp2_tp2_dp2_mtp"],
        gpu_type,
        hybrid = True,
        # Admit one fresh prefill per round while allowing concurrent in-flight requests.
        # MoE + EP4 exercises stage-local expert communication, including idle DP ranks.
        extra_args = " --max_batch_tokens_without_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0",
    )
    _pp_test(
        "qwen35_pd_mtp_pp_test",
        4,
        ["sym_mtp3", "pp2_pp1_mtp3"],
        gpu_type,
        hybrid = True,
    )

    # Reuse the main smoke topologies and model, adding long/short chunk requests.
    # Two recurrent-state blocks of budget leave room for a short final chunk.
    _pp_test(
        "qwen35_pp_chunk_prefill_test",
        4,
        ["pdfusion_pp4_tp1"],
        gpu_type,
        hybrid = True,
        chunk_size = 4096,
    )
    _pp_test(
        "qwen35_pp_chunk_prefill_mixed_mtp_test",
        8,
        ["pdfusion_pp2_tp2_dp2_mtp"],
        gpu_type,
        hybrid = True,
        extra_args = " --use_deepep_moe 1 --use_deepep_low_latency 0",
        chunk_size = 4096,
    )
    _pp_test(
        "qwen35_pd_pp_chunk_prefill_mtp_test",
        4,
        ["sym_mtp3", "pp2_pp1_mtp3"],
        gpu_type,
        hybrid = True,
        chunk_size = 4096,
    )

    native.test_suite(
        name = "smoke_pp_chunk_prefill",
        tests = [
            ":qwen35_pp_chunk_prefill_test",
            ":qwen35_pp_chunk_prefill_mixed_mtp_test",
            ":qwen35_pd_pp_chunk_prefill_mtp_test",
        ],
        tags = ["manual"],
    )

    native.test_suite(
        name = "smoke_pp",
        tests = [
            ":qwen35_pp_depth_test",
            ":qwen35_pp_mixed_mtp_test",
            ":qwen35_pd_mtp_pp_test",
            ":smoke_pp_chunk_prefill",
        ],
        tags = ["manual"],
    )
