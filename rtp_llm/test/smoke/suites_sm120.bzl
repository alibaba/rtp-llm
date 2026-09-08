load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm120_suites():
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            # Qwen3.5 uses independent pools for full-attention KV pages and
            # recurrent linear-attention states. Reusing one 64-token block
            # exercises the first GDN state boundary on the SM120 Triton path.
            smoke_test(
                name = "qwen35_dense_bf16_block64_reuse_sm120",
                task_info = "data/model/qwen35/qwen35_dense_bf16_block64_reuse_sm120.json",
                smoke_args = "--act_type BF16 --seq_size_per_block 64 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 4096 --tp_size 1 --world_size 1 --reuse_cache 1",
                gpu_type = ["RTX_5000_PRO"],
            ),
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
            smoke_test(
                name = "dense_fp8pt_dynamic_sm120",
                task_info = "data/model/qwen3/q_r_fp8pt_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16 --warm_up 0",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    # SM120 MoE FP8 (RTX 5000 Pro), auto strategy selects the DeepGEMM executor
    native.test_suite(
        name = "smoke_sm120_moe",
        tests = [
            smoke_test(
                name = "moe_fp8pb_tp2_sm120",
                task_info = "data/model/qwen3_moe/q_r_30b_fp8pb_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--moe_strategy auto --quantization FP8_PER_BLOCK --warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )
