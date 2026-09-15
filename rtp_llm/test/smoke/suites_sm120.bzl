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
                name = "generation_prefill_cuda_graph_sm120",
                task_info = "data/model/qwen25/q_r_generation_prefill_cuda_graph_sm120.json",
                # Keep framework warmup enabled and let it size the production
                # KV pool. This gates graph capture memory accounting instead of
                # bypassing it with a fixed test_block_num.
                smoke_args = "--act_type BF16 --warm_up 1 --seq_size_per_block 64 --concurrency_limit 5 --max_context_batch_size 5 --enable_cuda_graph 1 --decode_capture_config '1' --generation_prefill_cuda_graph_max_requests 5 --generation_prefill_capture_config '64,256'",
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
            smoke_test(
                name = "embedding_bert_sm120",
                task_info = "data/model/bert/q_r.json",
                smoke_args = "--seq_size_per_block 16 --act_type FP16",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    # SM120 dense FP8; FP8_PER_BLOCK is routed through the shared DeepGEMM path.
    native.test_suite(
        name = "smoke_sm120_dense",
        tests = [
            smoke_test(
                name = "dense_fp8pb_dynamic_sm120",
                task_info = "data/model/qwen3/q_r_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    # Qwen3.5 MoE: quantize BF16 weights to FP8 per-block at load time.
    native.test_suite(
        name = "smoke_sm120_moe",
        tests = [
            smoke_test(
                name = "moe_fp8pb_tp2_sm120",
                task_info = "data/model/qwen35/q_r_35b_fp8pb_tp2_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--moe_strategy auto --quantization FP8_PER_BLOCK --warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 2048 --concurrency_limit 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )
