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

    native.test_suite(
        name = "smoke_sm120_qwen4_exp",
        tests = [
            _qwen4_scenarios("qwen4_exp_mixed_tp2_eager_sm120", 0, False),
            _qwen4_scenarios("qwen4_exp_mixed_tp2_mtp_eager_sm120", 0, True),
            _qwen4_scenarios("qwen4_exp_mixed_tp2_mtp_graph_sm120", 1, True),
        ],
    )

def _qwen4_scenarios(name, graph, mtp):
    return custom_smoke_test(
        name = name,
        main = "qwen4_exp_smoke_test.py",
        smoke_args = _QWEN4_SMOKE_ARGS + (" --sp_type mtp --sp_model_type qwen4_exp_mtp --sp_act_type BF16 --gen_num_per_cycle 3" if mtp else "") + " --enable_cuda_graph " + str(graph),
        gpu_type = ["RTX_5000_PRO_CU13"],
        data = ["//rtp_llm:sdk", ":smoke_framework_srcs", "data/model/qwen4_exp/q_r_qwen4_exp_fp8_full.json"],
        deps = ["//rtp_llm:transformers"],
    )

_QWEN4_SMOKE_ARGS = "--load_method scratch --act_type BF16 --tp_size 2 --dp_size 1 --ep_size 2 --ffn_sp_size 2 --world_size 2 --reserver_runtime_mem_mb 4096 --max_seq_len 32768 --seq_size_per_block 128 --kernel_seq_size_per_block 128 --fp8_kv_cache 1 --quantization QWEN4_FP8_NVFP4 --moe_strategy nvfp4_cutlass --fp4_moe_op cutlass --use_deepep_moe 0 --enable_fp32_lm_head 0 --reuse_cache 1"
