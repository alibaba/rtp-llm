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
            smoke_test(
                name = "embedding_bert_sm120",
                task_info = "data/model/bert/q_r.json",
                smoke_args = "--seq_size_per_block 16 --act_type FP16",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )

    # SM120 MoE NVFP4 coverage on RTX 5000 Pro.
    native.test_suite(
        name = "smoke_sm120_moe",
        tests = [
            # Golden tokens are checkpoint-specific; a different NVFP4 weight
            # revision or conversion may legitimately fail response comparison.
            # Keep a two-token probe so this smoke exercises prefill plus decode;
            # executor tests own deeper numerical/Graph coverage.
            # Keep generic warmup disabled; the first smoke request exercises JIT.
            smoke_test(
                name = "moe_nvfp4_no_deepep_sm120",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--moe_strategy auto --fp4_moe_op auto --use_deepep_moe 0 --use_all_gather 1 --warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "moe_nvfp4_tp2_sm120",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = "--moe_strategy auto --fp4_moe_op auto --use_deepep_moe 0 --use_all_gather 1 --warm_up 0 --act_type BF16 --tp_size 2 --ep_size 1 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO"],
            ),
            smoke_test(
                name = "moe_nvfp4_async_tp2_sm120",
                task_info = "data/model/qwen3_moe/q_r_coder_30b_nvfp4_sm120.json",
                envs = [
                    "LOAD_PYTHON_MODEL=1",
                    "RTP_LLM_STREAM_ASYNC=1",
                    "RTP_LLM_DROP_BROAD_SYNC=1",
                    "RTP_LLM_DEVICE_INPUT=1",
                    "RTP_LLM_DEVICE_INPUT_CHECK=1",
                ],
                smoke_args = "--moe_strategy auto --fp4_moe_op auto --use_deepep_moe 0 --use_all_gather 1 --warm_up 0 --act_type BF16 --tp_size 2 --ep_size 1 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO"],
            ),
        ],
    )
