load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm120_suites():
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            smoke_test(
                name = "softmax_probs_sm120",
                task_info = "data/model/qwen25/q_r_softmax_probs_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "fp16_sm120",
                task_info = "data/model/qwen25/q_r_s_fp16_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "bf16_sm120",
                task_info = "data/model/qwen25/q_r_s_bf16_sm120.json",
                smoke_args = "--act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "bf16_cuda_graph_sm120",
                task_info = "data/model/qwen25/q_r_s_bf16_sm120.json",
                smoke_args = "--act_type BF16 --warm_up 0 --seq_size_per_block 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "random_seed_sm120",
                task_info = "data/model/qwen25/test_random_seed_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "logits_index_sm120",
                task_info = "data/model/qwen25/logits_index_q_r_sm120.json",
                smoke_args = "--act_type FP16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    # SM120 MoE: single-GPU FlashInfer B12X NVFP4 coverage.
    native.test_suite(
        name = "smoke_sm120_moe",
        tests = [
            # Golden tokens are checkpoint-specific; a different NVFP4 weight
            # revision or conversion may legitimately fail response comparison.
            # Keep the response probe to one token because low-precision greedy
            # suffixes can bifurcate; executor tests own numerical/Graph checks.
            # Keep generic warmup disabled; the first smoke request exercises JIT.
            smoke_test(
                name = "moe_nvfp4_no_deepep_sm120",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1", "STABILITY_REPEAT=5"],
                smoke_args = "--moe_strategy fp4_b12x --fp4_moe_op b12x --use_deepep_moe 0 --use_all_gather 1 --warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )
