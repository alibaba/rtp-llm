load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def h20_oss_suites():
    # H20 (SM9x) — Architecture-grouped suites
    # ============================================================================

    # H20 MLA (DeepSeek V2/V3.2, GLM-5)
    native.test_suite(
        name = "smoke_h20_mla",
        tests = [
            smoke_test(
                name="mla_fp8_redundant_expert_tp2",
                task_info="data/model/deepseek_v2/q_r_3090_mla_r24.json",
                smoke_args="--masked_max_token_num 0 --redundant_expert 24 --act_type BF16 --quantization FP8_PER_BLOCK --tp_size 2 --world_size 2",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_fp8_reuse_absorb_tp2",
                task_info="data/model/deepseek_v2/q_r_3090_mla.json",
                smoke_args="--load_method scratch --reuse_cache 1 --seq_size_per_block 8 --act_type BF16 --quantization FP8_PER_BLOCK --absorb_opt_len 1 --tp_size 2 --world_size 2",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_noquant_dp2",
                task_info="data/model/deepseek_v2/q_r_mla_pymodel.json",
                smoke_args="--seq_size_per_block 8 --act_type BF16 --tp_size 1 --dp_size 2 --world_size 2 --reserver_runtime_mem_mb 16697",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_cudagraph_pad_reuse",
                task_info="data/model/deepseek_v2/q_r_mla_pymodel.json",
                smoke_args="--warm_up 0 --test_block_num 1000 --tp_size 1 --world_size 1 --reuse_cache 1 --seq_size_per_block 64 --act_type BF16 --decode_capture_config '2' --enable_cuda_graph 1",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_cudagraph_fp8pt_deepep_dp2",
                task_info="data/model/deepseek_v2/q_r_mla_cudagraph_per_tensor.json",
                smoke_args="--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 1 --quantization FP8_DYNAMIC_PER_TENSOR --use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 1 --world_size 2 --dp_size 2 --reserver_runtime_mem_mb 16697",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_cudagraph_fp8pb_deepep_dp2",
                task_info="data/model/deepseek_v2/q_r_mla_pymodel_cudagraph.json",
                smoke_args="--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 1 --quantization FP8_PER_BLOCK --use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 1 --world_size 2 --dp_size 2 --reserver_runtime_mem_mb 16697",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_kernel_block_size",
                task_info="data/model/glm5/glm_5_fp8_q_r_h20.json",
                smoke_args="--warm_up 0 --seq_size_per_block 512 --act_type BF16 --enable_cuda_graph 0 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 1 --kernel_seq_size_per_block 64",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_cudagraph_deepep_tp2",
                task_info="data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_cuda_graph.json",
                smoke_args="--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 1 --reserver_runtime_mem_mb 20000 --tp_size 2 --world_size 2 --dp_size 1 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_fp8_basic",
                task_info="data/model/deepseek_v32_4layers/v32_fp8_q_r_h20.json",
                smoke_args="--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 0 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 1",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_not_fast_path_reuse",
                task_info="data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_long.json",
                smoke_args="--warm_up 0 --seq_size_per_block 64 --act_type BF16 --reserver_runtime_mem_mb 49343 --enable_cuda_graph 0 --reuse_cache 1 --hack_layer_num 1 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 1",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_fast_path_reuse",
                task_info="data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_66_seq.json",
                smoke_args="--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 0 --reuse_cache 1 --hack_layer_num 1 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 0",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="mla_cp_pd",
                task_info="data/model/glm5/glm_5_fp8_q_r_h20_cp.json",
                envs={
                    "prefill": [],
                    "decode": []},
                smoke_args={
                    "prefill": "--fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 8192 --role_type PREFILL --seq_size_per_block 64 --dp_size 1 --tp_size 2 --ep_size 2 --world_size 2 --warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --use_all_gather=0",
                    "decode": "--fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 8192 --role_type DECODE --seq_size_per_block 64 --ep_size 2 --dp_size 2 --world_size 2 --warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --cp_rotate_method PREFILL_CP --use_all_gather=0"
                },
                gpu_type=["H20"]
            ),
            smoke_test(
                name="mla_load_quant_tp2",
                task_info="data/model/deepseek-r1-4layer/r1_fp8_q_r_h20.json",
                smoke_args="--cache_store_rdma_mode 0 --use_local 1 --seq_size_per_block 64 --decode_entrance 1 --act_type bf16 --quantization FP8_PER_BLOCK --tp_size 2 --reserver_runtime_mem_mb 5026",
                gpu_type=["H20"]
            ),
        ],
    )


    # H20 MoE (Qwen3-30B-A3B)
    native.test_suite(
        name = "smoke_h20_moe",
        tests = [
            smoke_test(
                name="moe_masked_fp8_tp2",
                task_info="data/model/qwen3_moe/q_r_30b_py_masked_without_deepep_tp2.json",
                smoke_args="--moe_strategy fp8_per_block_no_dp_masked --quantization FP8_PER_BLOCK --warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="moe_w4a8_int4",
                task_info="data/model/qwen3_moe/q_r_30b_py_w4a8_int4_ptpc.json",
                smoke_args="--quantization W4A8_INT4_PER_CHANNEL --warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="moe_deepep_continuous_dp2",
                task_info="data/model/qwen3_moe/q_r_30b_py.json",
                smoke_args="--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 0 --dp_size 2",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="moe_deepep_ll_tp2",
                task_info="data/model/qwen3_moe/q_r_30b_py_tp2_ll.json",
                smoke_args="--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 2 --world_size 2",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="moe_deepep_normal_tp2",
                task_info="data/model/qwen3_moe/q_r_30b_py_tp2.json",
                smoke_args="--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size 2",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="moe_headwise",
                task_info="data/model/qwen3_moe/q_r_30b_py_headwise.json",
                envs=["WARM_UP=0", "LOAD_PYTHON_MODEL=1", "ACT_TYPE=BF16", "TP_SIZE=1", "WORLD_SIZE=1", "RESERVER_RUNTIME_MEM_MB=8192", "MAX_SEQ_LEN=32768"],
                smoke_args=["--use_deepep_moe 1", "--use_deepep_low_latency 0"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="moe_fp8pt_batched_cudagraph",
                task_info="data/model/qwen3_moe/q_r_fp8_per_tensor_batched_cuda_graph_30b.json",
                smoke_args="--decode_capture_config '1,2' --reserver_runtime_mem_mb 20000 --warm_up 0 --act_type BF16 --enable_cuda_graph 1  --use_deepep_moe 1 --use_deepep_low_latency 1 --dp_size 2 --ep_size 2",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="moe_fp8pt_load_quant",
                task_info="data/model/qwen3_moe/q_r_fp8_per_tensor_30b_load_quant.json",
                smoke_args="--warm_up 0 --act_type BF16 --quantization FP8_DYNAMIC_PER_TENSOR --use_deepep_moe 1 --use_deepep_low_latency 0 --force_cpu_load_weights 1 --dp_size 2 --ep_size 2",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="moe_cp_pd",
                task_info="data/model/qwen3_moe/q_r_30b_fp8_py_cp2.json",
                envs={
                    "prefill": [],
                    "decode": ["ACCL_LOW_LATENCY_OPTIMIZE=1"]},
                smoke_args={
                    "prefill": "--act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 8192 --role_type PREFILL --seq_size_per_block 64 --dp_size 1 --tp_size 2 --ep_size 2 --world_size 2 --warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER",
                    "decode": "--act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 8192 --role_type DECODE --seq_size_per_block 64 --ep_size 2 --dp_size 2 --world_size 2 --warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --cp_rotate_method PREFILL_CP",
                },
                gpu_type=["H20"],
            ),
        ],
    )


    # H20 Dense (Qwen2.5/Qwen3 dense)
    native.test_suite(
        name = "smoke_h20_dense",
        tests = [
            smoke_test(
                name="dense_fp8kv_cudagraph",
                task_info="data/model/qwen25/q_r_new_model_py_fp8_kv_cache.json",
                smoke_args="--warm_up 0 --seq_size_per_block 64 --act_type BF16 --test_block_num 1000 --fp8_kv_cache 1 --enable_cuda_graph 1  --disable_flash_infer 1",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="dense_fp8_prequant_tp2",
                task_info="data/model/qwen3/q_r_block_fp8.json",
                smoke_args="--disable_flash_infer 1 --act_type BF16 --reserver_runtime_mem_mb 8192 --tp_size 2 --warm_up 0",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="dense_fp8pb_dynamic",
                task_info="data/model/qwen3/q_r_h20.json",
                smoke_args="--disable_flash_infer 1 --quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="dense_fp8pt_dynamic",
                task_info="data/model/qwen3/q_r_h20_per_tensor_w13.json",
                smoke_args="--disable_flash_infer 1 --quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="dense_override_yarn",
                task_info="data/model/qwen3/q_r_override_yarn.json",
                smoke_args="--reserver_runtime_mem_mb 20000 --json_model_override_args '{\\\"rope_scaling\\\":{\\\"type\\\":\\\"yarn\\\",\\\"factor\\\":2.0,\\\"original_max_position_embeddings\\\":32768,\\\"beta_slow\\\":1.0,\\\"beta_fast\\\":1.0,\\\"mscale\\\":1.0,\\\"extrapolation_factor\\\":1.0}}' --seq_size_per_block 64 --act_type BF16 --warm_up 0",
                gpu_type=["H20"],
            ),
        ],
    )


    # H20 Qwen3.5/Next
    native.test_suite(
        name = "smoke_h20_next",
        tests = [
            smoke_test(
                name="next_mtp_basic",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2_mtp.json",
                smoke_args="--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_mtp_reuse",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2_mtp_reuse_cache.json",
                smoke_args="--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --reuse_cache 1",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_mtp_cudagraph_deepep",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2_mtp_cudagraph.json",
                smoke_args="--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --warm_up 0 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2,3,4' --use_deepep_moe 1 --use_deepep_low_latency 1",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_mtp_pd_reuse",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2_mtp_pd.json",
                smoke_args= {
                    "prefill": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 --act_type BF16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --reuse_cache 1",
                    "decode": "--load_cache_timeout_ms 120000 --act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --warm_up 0 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2,3,4' --use_deepep_moe 1 --use_deepep_low_latency 1 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1"
                },
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_fp8_basic",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2.json",
                smoke_args="--act_type BF16 --seq_size_per_block 2048 --tp_size 2",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_kernel_block",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2_kernel_block_size_128.json",
                smoke_args="--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --kernel_seq_size_per_block 128",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_cudagraph_deepep",
                task_info="data/model/qwen3_next/q_r_next_cuda_graph.json",
                smoke_args="--act_type BF16 --seq_size_per_block 2048 --max_seq_len 128 --use_deepep_moe 1 --use_deepep_low_latency 1 --enable_cuda_graph 1 --warm_up 0  --concurrency_limit 8 --reserver_runtime_mem_mb 8192 --tp_size 2",
                envs=["ACCL_LOW_LATENCY_OPTIMIZE=1"],
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_long_reuse_memcache",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2_long_input_reuse_cache.json",
                smoke_args="--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --linear_step 2 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_long_reuse_remote",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2_long_input_reuse_remote_cache.json",
                smoke_args="--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --linear_step 2 --reuse_cache 1 --enable_remote_cache 1 --write_cache_sync 1 --reco_put_timeout_ms 17000 --reco_get_timeout_ms 17000 --reco_get_broadcast_timeout 20000 --reco_put_broadcast_timeout 20000",
                gpu_type=["H20"],
                kvcm_envs = ["KVCM_LOG_LEVEL=DEBUG"],
                data = ["@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin"],
            ),
            smoke_test(
                name="next_bf16_basic",
                task_info="data/model/qwen35/qwen35_bf16_tp2.json",
                smoke_args="--tp_size 2 --act_type BF16 --seq_size_per_block 2048",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_load_quant_tp2",
                task_info="data/model/qwen35/qwen35_bf16_tp2_load_quant.json",
                smoke_args="--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --quantization fp8_per_block",
                gpu_type=["H20"],
            ),
            smoke_test(
                name="next_pd",
                task_info="data/model/qwen3_next/q_r_next_fp8_tp2_pd_sep.json",
                smoke_args={
                    "prefill": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 --act_type BF16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --tp_size 2 --reserver_runtime_mem_mb 9861 --ssm_state_dtype fp32",
                    "decode": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 --act_type BF16 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --tp_size 2 --reserver_runtime_mem_mb 9861 --ssm_state_dtype fp32"
                },
                gpu_type=["H20"],
            ),
        ],
    )


    # H20 Eagle (Qwen2-14B + draft model)
    native.test_suite(
        name = "smoke_h20_eagle",
        tests = [
            smoke_test(
                name="eagle_mtp_tp2",
                task_info="data/model/qwen2_14b/q_r_mtp.json",
                smoke_args="--max_seq_len 16384 --ft_disable_custom_ar 1 --sp_type eagle --gen_num_per_cycle 4 --act_type FP16 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/  --warm_up 0 --reserver_runtime_mem_mb 21954 --tp_size 2",
                gpu_type=["H20"]
            ),
            smoke_test(
                name="eagle_mtp_reuse",
                task_info="data/model/qwen2_14b/q_r_mtp_reuse_cache.json",
                smoke_args="--reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1 --max_seq_len 16384 --ft_disable_custom_ar 1 --sp_type eagle --gen_num_per_cycle 4 --act_type FP16 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/  --warm_up 0 --reserver_runtime_mem_mb 21954 --tp_size 2",
                gpu_type=["H20"]
            ),
            smoke_test(
                name="eagle_mtp_cudagraph",
                task_info="data/model/qwen2_14b/q_r_mtp_cudagraph.json",
                smoke_args="--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode NONE --redundant_expert 0 --act_type FP16 --concurrency_limit 64 --frontend_server_count 1 --warm_up 0 --reserver_runtime_mem_mb 24096 --seq_size_per_block 64 --enable_xqa 1 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 --decode_capture_config '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16'  --enable_cuda_graph 1 --load_method scratch --tp_size 1 --world_size 1 --dp_size 1",
                envs=["NCCL_DISABLE_ABORT=1", "NCCL_DEBUG=INFO", "LOG_LEVEL=INFO"],
                gpu_type=["H20"]
            ),
            smoke_test(
                name="eagle_mtp_cudagraph_concurrent",
                task_info="data/model/qwen2_14b/q_r_mtp_cuda_graph_concurrent.json",
                smoke_args="--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode NONE --redundant_expert 0 --act_type FP16 --concurrency_limit 16 --frontend_server_count 1 --warm_up 0 --reserver_runtime_mem_mb 42000 --seq_size_per_block 64 --enable_xqa 1 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 --decode_capture_config '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16' --prefill_capture_config '80:1' --enable_cuda_graph 1 --deterministic_attn 1 --tp_size 2",
                envs=["NCCL_DISABLE_ABORT=1", "NCCL_DEBUG=INFO", "LOG_LEVEL=INFO"],
                gpu_type=["H20"],
                concurrency_test=True,
            ),
            smoke_test(
                name="eagle_mtp_no_cudagraph_concurrent",
                task_info="data/model/qwen2_14b/q_r_mtp_cuda_graph_concurrent.json",
                smoke_args="--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode NONE --redundant_expert 0 --act_type FP16 --concurrency_limit 16 --frontend_server_count 1 --warm_up 0 --reserver_runtime_mem_mb 42000 --seq_size_per_block 64 --enable_xqa 1 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 --deterministic_attn 1 --tp_size 2",
                envs=["NCCL_DISABLE_ABORT=1", "NCCL_DEBUG=INFO", "LOG_LEVEL=INFO"],
                gpu_type=["H20"],
                concurrency_test=True,
            ),
            smoke_test(
                name="eagle_remote_cache_tp2",
                task_info="data/model/qwen_sp/q_r_remote_cache_sp_tpsize2.json",
                data=["@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin"],
                kvcm_envs=["KVCM_LOG_LEVEL=DEBUG"],
                sleep_time_qr=20,
                smoke_args="--warm_up 0 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --tp_size 2 --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --act_type FP16 --reuse_cache 1 --seq_size_per_block 8 --max_seq_len 16384 --ft_disable_custom_ar 1 --warm_up 0 --reserver_runtime_mem_mb 21954 --test_block_num 500 --enable_remote_cache true --enable_device_cache 0 --enable_memory_cache 0 --deterministic_attn 1 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000",
                gpu_type=["H20"],
            ),
        ],
    )

