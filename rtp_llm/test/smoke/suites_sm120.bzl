load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test", "smoke_test")

def sm120_suites():
    native.test_suite(
        name = "smoke_sm120_basic",
        tests = [
            custom_smoke_test(
                name = "qwen3_bf16_sm120",
                main = "sm120_qwen3_test.py",
                # One PDFUSION server runs all generation/API/cache scenarios.
                smoke_args = "--role_type PDFUSION --act_type BF16 --warm_up 0 --tp_size 1 --world_size 1 --frontend_server_count 1 --max_seq_len 8192 --seq_size_per_block 64 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2' --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0",
                data = [
                    "//rtp_llm:sdk",
                    "data/model/qwen3/q_r_1_7b_bf16_sm120.json",
                ],
                deps = ["//rtp_llm:transformers"],
                gpu_type = ["RTX_5000_PRO_CU13"],
            ),
            smoke_test(
                name = "generation_prefill_cuda_graph_sm120",
                task_info = "data/model/qwen25/q_r_generation_prefill_cuda_graph_sm120.json",
                # Keep framework warmup enabled and let it size the production
                # KV pool. This gates graph capture memory accounting instead of
                # bypassing it with a fixed test_block_num.
                smoke_args = "--act_type BF16 --warm_up 1 --seq_size_per_block 64 --concurrency_limit 5 --max_context_batch_size 5 --enable_cuda_graph 1 --decode_capture_config '1' --generation_prefill_cuda_graph_max_requests 5 --generation_prefill_capture_config '64,256'",
                gpu_type = ["RTX_5000_PRO_CU13"],
            ),
            smoke_test(
                name = "embedding_bert_sm120",
                task_info = "data/model/bert/q_r.json",
                smoke_args = "--seq_size_per_block 16 --act_type FP16",
                gpu_type = ["RTX_5000_PRO_CU13"],
            ),
        ],
    )

    # Quantize Qwen3 dense BF16 weights at load time on two SM120 devices.
    dense_args = "--quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0 --tp_size 2 --world_size 2 --seq_size_per_block 2048 --max_seq_len 16384 --reserver_runtime_mem_mb 16005 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2'"
    native.test_suite(
        name = "smoke_sm120_dense",
        tests = [
            smoke_test(
                name = "dense_fp8pb_dynamic_sm120",
                task_info = "data/model/qwen3/q_r_1_7b_fp8pb_tp2_sm120.json",
                envs = ["LOAD_PYTHON_MODEL=1"],
                smoke_args = dense_args + " --reuse_cache 0",
                gpu_type = ["RTX_5000_PRO_CU13"],
            ),
            custom_smoke_test(
                name = "dense_fp8pb_reuse_cache_tp2_sm120",
                main = "sm120_reuse_cache_test.py",
                smoke_args = dense_args + " --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0",
                data = ["//rtp_llm:sdk"],
                deps = ["//rtp_llm:transformers"],
                gpu_type = ["RTX_5000_PRO_CU13"],
            ),
        ],
    )

    # Qwen3.5 MoE: quantize BF16 weights to FP8 per-block at load time.
    native.test_suite(
        name = "smoke_sm120_moe",
        tests = [
            custom_smoke_test(
                name = "moe_fp8pb_tp2_sm120",
                main = "sm120_moe_test.py",
                data = [
                    "//rtp_llm:sdk",
                    ":smoke_framework_srcs",
                    "data/model/qwen35/q_r_35b_fp8pb_tp2_sm120.json",
                    "data/prompt_candidates.json",
                ],
                smoke_args = "--moe_strategy auto --quantization FP8_PER_BLOCK --warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 2048 --concurrency_limit 64 --enable_cuda_graph 1 --decode_capture_config '1,2'",
                gpu_type = ["RTX_5000_PRO_CU13"],
            ),
        ],
    )
