load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def sm100_suites():
    # SM100 / GB200 (SM100_ARM)
    # ============================================================================

    # SM100 Dense (Qwen3 dense + FP8 attention)
    native.test_suite(
        name = "smoke_sm100_dense",
        tests = [
            smoke_test(
                name="dense_tp1_sm100",
                task_info="data/model/qwen3/q_r_l20a_fp4_tp1_py.json",
                smoke_args="--warm_up 0 --act_type BF16 --tp_size 1 --world_size 1 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
                gpu_type=["SM100_ARM"],
            ),
            smoke_test(
                name="dense_tp2_sm100",
                task_info="data/model/qwen3/q_r_l20a_fp4_tp2_py.json",
                smoke_args="--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
                gpu_type=["SM100_ARM"],
            ),
            smoke_test(
                name="fp8_attention_sm100",
                task_info="data/model/qwen3/q_r_block_fp8.json",
                smoke_args="--act_type BF16 --seq_size_per_block 64 --fp8_kv_cache 1 --reserver_runtime_mem_mb 178125 --warm_up 0",
                gpu_type=["SM100_ARM"],
            ),
        ],
    )


    # SM100 MoE (Qwen3-30B MoE + Qwen3.5 MoE on SM100)
    native.test_suite(
        name = "smoke_sm100_moe",
        tests = [
            # REMOVED: moe_deepep_ll_dp2_sm100 (FP8_PER_BLOCK + DeepEP LL + DP2)
            # Reason: DeepEP LL path already covered by 5 NVFP4 LL cases; UE8M0 scale path
            # covered by moe_deepep_normal_tp2_sm100; H20 has moe_deepep_ll_tp2 for LL+FP8.
            smoke_test(
                name="moe_deepep_normal_tp2_sm100",
                task_info="data/model/qwen3_moe/q_r_30b_py_tp2_sm100.json",
                smoke_args="--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --quantization FP8_PER_BLOCK --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size 2",
                gpu_type=["SM100_ARM"],
            ),
            # REMOVED: moe_nvfp4_deepep_ll_dp2_sm100 (NVFP4 + DeepEP LL + DP2, no CudaGraph)
            # Reason: moe_nvfp4_deepep_ll_cudagraph_dp2_sm100 is its CudaGraph superset
            # (same config + enable_cuda_graph), per Graph覆盖原则.
            smoke_test(
                name="moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
                task_info="data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_ll_cg_sm100_arm.json",
                smoke_args="--decode_capture_config '1,2' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type=["SM100_ARM"],
            ),
            smoke_test(
                name="moe_nvfp4_deepep_ll_tp2_sm100",
                task_info="data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_ll_sm100_arm.json",
                smoke_args="--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type=["SM100_ARM"],
            ),
            smoke_test(
                name="moe_nvfp4_deepep_normal_dp2_sm100",
                task_info="data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_normal_sm100_arm.json",
                smoke_args="--warm_up 0 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0",
                gpu_type=["SM100_ARM"],
            ),
            smoke_test(
                name="moe_nvfp4_deepep_normal_tp2_sm100",
                task_info="data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_normal_sm100_arm.json",
                smoke_args="--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0",
                gpu_type=["SM100_ARM"],
            ),
            smoke_test(
                name="next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100",
                task_info="data/model/qwen35/q_r_35b_nvfp4_py_dp2_ll_cg_sm100_arm.json",
                smoke_args="--decode_capture_config '1,2,3,4' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --seq_size_per_block 2048 --concurrency_limit 64  --kernel_seq_size_per_block 64 --use_deepep_moe 1 --use_deepep_low_latency 1",
                gpu_type=["SM100_ARM"],
            ),
            smoke_test(
                name="next_moe_nvfp4_cudagraph_tp2_sm100",
                task_info="data/model/qwen35/q_r_35b_nvfp4_py_tp2_cg_sm100_arm.json",
                smoke_args="--decode_capture_config '1,2,3,4' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --seq_size_per_block 2048 --concurrency_limit 64  --kernel_seq_size_per_block 64",
                gpu_type=["SM100_ARM"],
            ),
            # REMOVED: moe_masked_cudagraph_tp2_sm100 (Masked MoE + FP8_PER_BLOCK + CudaGraph + TP2)
            # Reason: shares task_info with H20 moe_masked_fp8_tp2 (q_r_30b_py_masked_without_deepep_tp2.json),
            # only differs by enable_cuda_graph and mem size; SM100 DeepGemm UE8M0 masked path
            # already covered by moe_deepep_normal_tp2_sm100 (same FP8_PER_BLOCK quantization).
        ]
    )

