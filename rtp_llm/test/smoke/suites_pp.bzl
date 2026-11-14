"""PP smoke: basic execution, mixed parallelism/MTP and both PD decode engines."""

load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

_COMMON_ARGS = "--act_type BF16 --warm_up 0 --enable_cuda_graph 0 --frontend_server_count 1 --concurrency_limit 4 --load_method scratch --shutdown_timeout 10"
# Qwen3.5 recurrent state uses 2048-token blocks; attention uses 64-token pages.
_HYBRID_ARGS = " --model_warm_up 0 --ssm_state_dtype fp32 --kv_cache_mem_mb 8192 --kernel_seq_size_per_block 64"

def _pp_test(name, gpus, cases, gpu_type, hybrid = False, extra_args = ""):
    custom_smoke_test(
        name = name,
        main = "pp_test.py",
        args = ["--cases=" + ",".join(cases)],
        smoke_args = "%s --world_size %d --max_seq_len 4096 --seq_size_per_block %d%s%s" % (
            _COMMON_ARGS,
            gpus,
            2048 if hybrid else 16,
            _HYBRID_ARGS if hybrid else "",
            extra_args,
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

    native.test_suite(
        name = "smoke_pp",
        tests = [
            ":qwen35_pp_depth_test",
            ":qwen35_pp_mixed_mtp_test",
            ":qwen35_pd_mtp_pp_test",
        ],
        tags = ["manual"],
    )
