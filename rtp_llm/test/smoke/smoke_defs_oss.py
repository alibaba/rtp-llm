"""
Unified Smoke Test Definitions.

Auto-generated from internal_source/rtp_llm/test/smoke/BUILD.
Regenerate with: python scripts/gen_smoke_defs.py BUILD > smoke_defs.py

Format matches the BUILD smoke_test() macro interface:
    smoke_args: CLI args string or dict (multi-role)
    envs: extra env vars list or dict (optional)
"""

SMOKE_TESTS = {
    "smoke_h20_mla": {
        "mla_fp8_redundant_expert_tp2": {
            "task_info": "data/model/deepseek_v2/q_r_3090_mla_r24.json",
            "smoke_args": "--masked_max_token_num 0 --redundant_expert 24 --act_type BF16 --quantization FP8_PER_BLOCK --tp_size 2 --world_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_fp8_reuse_absorb_tp2": {
            "task_info": "data/model/deepseek_v2/q_r_3090_mla.json",
            "smoke_args": "--load_method scratch --reuse_cache 1 --seq_size_per_block 8 --act_type BF16 --quantization FP8_PER_BLOCK --absorb_opt_len 1 --tp_size 2 --world_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_noquant_dp2": {
            "task_info": "data/model/deepseek_v2/q_r_mla_pymodel.json",
            "smoke_args": "--seq_size_per_block 8 --act_type BF16 --tp_size 1 --dp_size 2 --world_size 2 --reserver_runtime_mem_mb 16697",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_cudagraph_pad_reuse": {
            "task_info": "data/model/deepseek_v2/q_r_mla_pymodel.json",
            "smoke_args": "--warm_up 0 --test_block_num 1000 --tp_size 1 --world_size 1 --reuse_cache 1 --seq_size_per_block 64 --act_type BF16 --decode_capture_config '2' --enable_cuda_graph 1",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_cudagraph_fp8pt_deepep_dp2": {
            "task_info": "data/model/deepseek_v2/q_r_mla_cudagraph_per_tensor.json",
            "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 1 --quantization FP8_DYNAMIC_PER_TENSOR --use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 1 --world_size 2 --dp_size 2 --reserver_runtime_mem_mb 16697",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_cudagraph_fp8pb_deepep_dp2": {
            "task_info": "data/model/deepseek_v2/q_r_mla_pymodel_cudagraph.json",
            "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 1 --quantization FP8_PER_BLOCK --use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 1 --world_size 2 --dp_size 2 --reserver_runtime_mem_mb 16697",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_kernel_block_size": {
            "task_info": "data/model/glm5/glm_5_fp8_q_r_h20.json",
            "smoke_args": "--warm_up 0 --seq_size_per_block 512 --act_type BF16 --enable_cuda_graph 0 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 1 --kernel_seq_size_per_block 64",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_cudagraph_deepep_tp2": {
            "task_info": "data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_cuda_graph.json",
            "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 1 --reserver_runtime_mem_mb 20000 --tp_size 2 --world_size 2 --dp_size 1 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_fp8_basic": {
            "task_info": "data/model/deepseek_v32_4layers/v32_fp8_q_r_h20.json",
            "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 0 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 1",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_not_fast_path_reuse": {
            "task_info": "data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_long.json",
            "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --reserver_runtime_mem_mb 49343 --enable_cuda_graph 0 --reuse_cache 1 --hack_layer_num 1 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 1",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_fast_path_reuse": {
            "task_info": "data/model/deepseek_v32_4layers/v32_fp8_q_r_h20_66_seq.json",
            "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --enable_cuda_graph 0 --reuse_cache 1 --hack_layer_num 1 --tp_size 1 --world_size 1 --dp_size 1 --fp8_kv_cache 0",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_cp_pd": {
            "task_info": "data/model/glm5/glm_5_fp8_q_r_h20_cp.json",
            "smoke_args": {
                "prefill": "--fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 8192 --role_type PREFILL --seq_size_per_block 64 --dp_size 1 --tp_size 2 --ep_size 2 --world_size 2 --warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --use_all_gather=0",
                "decode": "--fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 8192 --role_type DECODE --seq_size_per_block 64 --ep_size 2 --dp_size 2 --world_size 2 --warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --cp_rotate_method PREFILL_CP --use_all_gather=0",
            },
            "envs": {
                "prefill": [],
                "decode": [],
            },
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "mla_load_quant_tp2": {
            "task_info": "data/model/deepseek-r1-4layer/r1_fp8_q_r_h20.json",
            "smoke_args": "--cache_store_rdma_mode 0 --use_local 1 --seq_size_per_block 64 --decode_entrance 1 --act_type bf16 --quantization FP8_PER_BLOCK --tp_size 2 --reserver_runtime_mem_mb 5026",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
    },
    "smoke_h20_moe": {
        "moe_masked_fp8_tp2": {
            "task_info": "data/model/qwen3_moe/q_r_30b_py_masked_without_deepep_tp2.json",
            "smoke_args": "--moe_strategy fp8_per_block_no_dp_masked --quantization FP8_PER_BLOCK --warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "moe_w4a8_int4": {
            "task_info": "data/model/qwen3_moe/q_r_30b_py_w4a8_int4_ptpc.json",
            "smoke_args": "--quantization W4A8_INT4_PER_CHANNEL --warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 16005 --seq_size_per_block 64 --concurrency_limit 64",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "moe_deepep_continuous_dp2": {
            "task_info": "data/model/qwen3_moe/q_r_30b_py.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 0 --dp_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "moe_deepep_ll_tp2": {
            "task_info": "data/model/qwen3_moe/q_r_30b_py_tp2_ll.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 1 --tp_size 2 --world_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "moe_deepep_normal_tp2": {
            "task_info": "data/model/qwen3_moe/q_r_30b_py_tp2.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "moe_headwise": {
            "task_info": "data/model/qwen3_moe/q_r_30b_py_headwise.json",
            "smoke_args": "",
            "envs": [
                "WARM_UP=0",
                "LOAD_PYTHON_MODEL=1",
                "ACT_TYPE=BF16",
                "TP_SIZE=1",
                "WORLD_SIZE=1",
                "RESERVER_RUNTIME_MEM_MB=8192",
                "MAX_SEQ_LEN=32768",
            ],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "moe_fp8pt_batched_cudagraph": {
            "task_info": "data/model/qwen3_moe/q_r_fp8_per_tensor_batched_cuda_graph_30b.json",
            "smoke_args": "--decode_capture_config '1,2' --reserver_runtime_mem_mb 20000 --warm_up 0 --act_type BF16 --enable_cuda_graph 1  --use_deepep_moe 1 --use_deepep_low_latency 1 --dp_size 2 --ep_size 2",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "moe_fp8pt_load_quant": {
            "task_info": "data/model/qwen3_moe/q_r_fp8_per_tensor_30b_load_quant.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --quantization FP8_DYNAMIC_PER_TENSOR --use_deepep_moe 1 --use_deepep_low_latency 0 --force_cpu_load_weights 1 --dp_size 2 --ep_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "moe_cp_pd": {
            "task_info": "data/model/qwen3_moe/q_r_30b_fp8_py_cp2.json",
            "smoke_args": {
                "prefill": "--act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 8192 --role_type PREFILL --seq_size_per_block 64 --dp_size 1 --tp_size 2 --ep_size 2 --world_size 2 --warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER",
                "decode": "--act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 8192 --role_type DECODE --seq_size_per_block 64 --ep_size 2 --dp_size 2 --world_size 2 --warm_up 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --cp_rotate_method PREFILL_CP",
            },
            "envs": {
                "prefill": [],
                "decode": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            },
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
    },
    "smoke_h20_dense": {
        "dense_fp8kv_cudagraph": {
            "task_info": "data/model/qwen25/q_r_new_model_py_fp8_kv_cache.json",
            "smoke_args": "--warm_up 0 --seq_size_per_block 64 --act_type BF16 --test_block_num 1000 --fp8_kv_cache 1 --enable_cuda_graph 1  --disable_flash_infer 1",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "dense_fp8_prequant_tp2": {
            "task_info": "data/model/qwen3/q_r_block_fp8.json",
            "smoke_args": "--disable_flash_infer 1 --act_type BF16 --reserver_runtime_mem_mb 8192 --tp_size 2 --warm_up 0",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "dense_fp8pb_dynamic": {
            "task_info": "data/model/qwen3/q_r_h20.json",
            "smoke_args": "--disable_flash_infer 1 --quantization FP8_PER_BLOCK --act_type BF16 --warm_up 0",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "dense_fp8pt_dynamic": {
            "task_info": "data/model/qwen3/q_r_h20_per_tensor_w13.json",
            "smoke_args": "--disable_flash_infer 1 --quantization FP8_DYNAMIC_PER_TENSOR --act_type BF16",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "dense_override_yarn": {
            "task_info": "data/model/qwen3/q_r_override_yarn.json",
            "smoke_args": "--reserver_runtime_mem_mb 20000 --json_model_override_args '{\\\\\\",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
    },
    "smoke_h20_next": {
        "next_mtp_basic": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_mtp.json",
            "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_mtp_reuse": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_mtp_reuse_cache.json",
            "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --reuse_cache 1",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_mtp_cudagraph_deepep": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_mtp_cudagraph.json",
            "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --warm_up 0 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2,3,4' --use_deepep_moe 1 --use_deepep_low_latency 1",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_mtp_pd_reuse": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_mtp_pd.json",
            "smoke_args": {
                "prefill": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 --act_type BF16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --reuse_cache 1",
                "decode": "--load_cache_timeout_ms 120000 --act_type BF16 --seq_size_per_block 2048 --tp_size 2 --max_seq_len 12800 --reserver_runtime_mem_mb 10000 --warm_up 0 --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_type eagle --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 --sp_act_type bf16 --concurrency_limit 4 --enable_cuda_graph 1 --decode_capture_config '1,2,3,4' --use_deepep_moe 1 --use_deepep_low_latency 1 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1",
            },
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_fp8_basic": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2.json",
            "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_kernel_block": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_kernel_block_size_128.json",
            "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --tp_size 2 --kernel_seq_size_per_block 128",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_cudagraph_deepep": {
            "task_info": "data/model/qwen3_next/q_r_next_cuda_graph.json",
            "smoke_args": "--act_type BF16 --seq_size_per_block 2048 --max_seq_len 128 --use_deepep_moe 1 --use_deepep_low_latency 1 --enable_cuda_graph 1 --warm_up 0  --concurrency_limit 8 --reserver_runtime_mem_mb 8192 --tp_size 2",
            "envs": ["ACCL_LOW_LATENCY_OPTIMIZE=1"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_long_reuse_memcache": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_long_input_reuse_cache.json",
            "smoke_args": "--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --linear_step 2 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_long_reuse_remote": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_long_input_reuse_remote_cache.json",
            "smoke_args": "--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --linear_step 2 --reuse_cache 1 --enable_remote_cache 1 --write_cache_sync 1 --reco_put_timeout_ms 17000 --reco_get_timeout_ms 17000 --reco_get_broadcast_timeout 20000 --reco_put_broadcast_timeout 20000",
            "envs": ["KVCM_LOG_LEVEL=DEBUG"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_bf16_basic": {
            "task_info": "data/model/qwen35/qwen35_bf16_tp2.json",
            "smoke_args": "--tp_size 2 --act_type BF16 --seq_size_per_block 2048",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_load_quant_tp2": {
            "task_info": "data/model/qwen35/qwen35_bf16_tp2_load_quant.json",
            "smoke_args": "--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --quantization fp8_per_block",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "next_pd": {
            "task_info": "data/model/qwen3_next/q_r_next_fp8_tp2_pd_sep.json",
            "smoke_args": {
                "prefill": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 --act_type BF16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --tp_size 2 --reserver_runtime_mem_mb 9861 --ssm_state_dtype fp32",
                "decode": "--load_cache_timeout_ms 120000 --seq_size_per_block 2048 --act_type BF16 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --tp_size 2 --reserver_runtime_mem_mb 9861 --ssm_state_dtype fp32",
            },
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
    },
    "smoke_h20_eagle": {
        "eagle_mtp_tp2": {
            "task_info": "data/model/qwen2_14b/q_r_mtp.json",
            "smoke_args": "--max_seq_len 16384 --ft_disable_custom_ar 1 --sp_type eagle --gen_num_per_cycle 4 --act_type FP16 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/  --warm_up 0 --reserver_runtime_mem_mb 21954 --tp_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "eagle_mtp_reuse": {
            "task_info": "data/model/qwen2_14b/q_r_mtp_reuse_cache.json",
            "smoke_args": "--reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1 --max_seq_len 16384 --ft_disable_custom_ar 1 --sp_type eagle --gen_num_per_cycle 4 --act_type FP16 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/  --warm_up 0 --reserver_runtime_mem_mb 21954 --tp_size 2",
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "eagle_mtp_cudagraph": {
            "task_info": "data/model/qwen2_14b/q_r_mtp_cudagraph.json",
            "smoke_args": "--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode NONE --redundant_expert 0 --act_type FP16 --concurrency_limit 64 --frontend_server_count 1 --warm_up 0 --reserver_runtime_mem_mb 24096 --seq_size_per_block 64 --enable_xqa 1 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 --decode_capture_config '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16'  --enable_cuda_graph 1 --load_method scratch --tp_size 1 --world_size 1 --dp_size 1",
            "envs": [
                "NCCL_DISABLE_ABORT=1",
                "NCCL_DEBUG=INFO",
                "LOG_LEVEL=INFO",
            ],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
        },
        "eagle_mtp_cudagraph_concurrent": {
            "task_info": "data/model/qwen2_14b/q_r_mtp_cuda_graph_concurrent.json",
            "smoke_args": "--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode NONE --redundant_expert 0 --act_type FP16 --concurrency_limit 16 --frontend_server_count 1 --warm_up 0 --reserver_runtime_mem_mb 42000 --seq_size_per_block 64 --enable_xqa 1 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 --decode_capture_config '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16' --prefill_capture_config '80:1' --enable_cuda_graph 1 --tp_size 2",
            "envs": [
                "NCCL_DISABLE_ABORT=1",
                "NCCL_DEBUG=INFO",
                "LOG_LEVEL=INFO",
            ],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
            "concurrency_test": True,
        },
        "eagle_mtp_no_cudagraph_concurrent": {
            "task_info": "data/model/qwen2_14b/q_r_mtp_cuda_graph_concurrent.json",
            "smoke_args": "--max_seq_len 16384 --ft_disable_custom_ar 1 --eplb_mode NONE --redundant_expert 0 --act_type FP16 --concurrency_limit 16 --frontend_server_count 1 --warm_up 0 --reserver_runtime_mem_mb 42000 --seq_size_per_block 64 --enable_xqa 1 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --sp_act_type FP16 --tp_size 2",
            "envs": [
                "NCCL_DISABLE_ABORT=1",
                "NCCL_DEBUG=INFO",
                "LOG_LEVEL=INFO",
            ],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
            "concurrency_test": True,
        },
        "eagle_remote_cache_tp2": {
            "task_info": "data/model/qwen_sp/q_r_remote_cache_sp_tpsize2.json",
            "smoke_args": "--warm_up 0 --sp_type eagle --gen_num_per_cycle 4 --sp_model_type qwen_2-mtp --tp_size 2 --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --act_type FP16 --reuse_cache 1 --seq_size_per_block 8 --max_seq_len 16384 --ft_disable_custom_ar 1 --warm_up 0 --reserver_runtime_mem_mb 21954 --test_block_num 500 --enable_remote_cache true --enable_device_cache 0 --enable_memory_cache 0 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000",
            "envs": ["KVCM_LOG_LEVEL=DEBUG"],
            "gpu_type": "H20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "H20"],
            "timeout": 600,
            "sleep_time_qr": 20,
        },
    },
    "smoke_sm8x_basic": {
        "softmax_probs": {
            "task_info": "data/model/qwen25/q_r_softmax_probs.json",
            "smoke_args": "--act_type FP16 --warm_up 0",
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "random_seed": {
            "task_info": "data/model/qwen25/test_random_seed.json",
            "smoke_args": "--act_type FP16 --warm_up 0",
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "fp16": {
            "task_info": "data/model/qwen25/q_r_s.json",
            "smoke_args": "--act_type FP16 --warm_up 0",
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "bf16": {
            "task_info": "data/model/qwen25/q_r_s.json",
            "smoke_args": "--act_type BF16 --warm_up 0",
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "tp2": {
            "task_info": "data/model/qwen25/q_r_s_fp16.json",
            "smoke_args": "--warm_up 0 --act_type FP16 --tp_size 2",
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "beam_search_tp2": {
            "task_info": "data/model/qwen25/bs_q_r.json",
            "smoke_args": "--act_type FP16 --tp_size 2 --warm_up 0",
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "frontend_app": {
            "task_info": "data/model/qwen25/q_r_3_front_app.json",
            "smoke_args": {
                "frontend": "--max_seq_len 2048 --role_type FRONTEND --warm_up 0",
                "pd_fusion": "--reuse_cache 1 --seq_size_per_block 8 --act_type FP16 --warm_up 0",
            },
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "logits_index": {
            "task_info": "data/model/qwen25/logits_index_q_r.json",
            "smoke_args": "--act_type FP16 --warm_up 0",
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "embedding_qwen_gte_7b_cudagraph": {
            "task_info": "data/model/qwen2/q_r_embedding.json",
            "smoke_args": "--seq_size_per_block 64 --embedding_model 1 --act_type BF16 --concurrency_limit 2 --enable_cuda_graph 1  --enable_cuda_graph_debug_mode 1 --prefill_capture_config '150,155,160,380,400' --task_type DENSE_EMBEDDING --reserver_runtime_mem_mb 3072",
            "envs": ["LOG_LEVEL=INFO", "PYTHONUNBUFFERED=TRUE"],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
    },
    "smoke_sm100_dense": {
        "dense_tp1_sm100": {
            "task_info": "data/model/qwen3/q_r_l20a_fp4_tp1_py.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --tp_size 1 --world_size 1 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
        "dense_tp2_sm100": {
            "task_info": "data/model/qwen3/q_r_l20a_fp4_tp2_py.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
        "fp8_attention_sm100": {
            "task_info": "data/model/qwen3/q_r_block_fp8.json",
            "smoke_args": "--act_type BF16 --seq_size_per_block 64 --fp8_kv_cache 1 --reserver_runtime_mem_mb 178125 --warm_up 0",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
    },
    "smoke_sm100_moe": {
        "moe_deepep_normal_tp2_sm100": {
            "task_info": "data/model/qwen3_moe/q_r_30b_py_tp2_sm100.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --quantization FP8_PER_BLOCK --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --tp_size 2 --world_size 2",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
        "moe_nvfp4_deepep_ll_cudagraph_dp2_sm100": {
            "task_info": "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_ll_cg_sm100_arm.json",
            "smoke_args": "--decode_capture_config '1,2' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
        "moe_nvfp4_deepep_ll_tp2_sm100": {
            "task_info": "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_ll_sm100_arm.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
        "moe_nvfp4_deepep_normal_dp2_sm100": {
            "task_info": "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_dp2_normal_sm100_arm.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
        "moe_nvfp4_deepep_normal_tp2_sm100": {
            "task_info": "data/model/qwen3_moe/q_r_coder_30b_nvfp4_py_tp2_normal_sm100_arm.json",
            "smoke_args": "--warm_up 0 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 8192 --fp8_kv_cache 1 --seq_size_per_block 64 --concurrency_limit 64 --blockwise_use_fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --use_all_gather 0",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
        "next_moe_nvfp4_deepep_ll_cudagraph_dp2_sm100": {
            "task_info": "data/model/qwen35/q_r_35b_nvfp4_py_dp2_ll_cg_sm100_arm.json",
            "smoke_args": "--decode_capture_config '1,2,3,4' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --dp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --seq_size_per_block 2048 --concurrency_limit 64  --kernel_seq_size_per_block 64 --use_deepep_moe 1 --use_deepep_low_latency 1",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
        "next_moe_nvfp4_cudagraph_tp2_sm100": {
            "task_info": "data/model/qwen35/q_r_35b_nvfp4_py_tp2_cg_sm100_arm.json",
            "smoke_args": "--decode_capture_config '1,2,3,4' --warm_up 0 --enable_cuda_graph 1 --act_type BF16 --tp_size 2 --world_size 2 --ep_size 2 --reserver_runtime_mem_mb 20000 --seq_size_per_block 2048 --concurrency_limit 64  --kernel_seq_size_per_block 64",
            "gpu_type": "SM100_ARM",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "SM100_ARM"],
            "timeout": 600,
        },
    },
    "smoke_rocm_basic": {
        "rocm_basic_cache_reuse": {
            "task_info": "data/model/qwen2/q_r_reuse.json",
            "smoke_args": "--reuse_cache 1 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_basic_batch_cache_reuse": {
            "task_info": "data/model/qwen3/q_r_308x_batch_cache.json",
            "smoke_args": "--reuse_cache 1 --enable_cuda_graph 1 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
    },
    "smoke_rocm_dense": {
        "rocm_dense_qwen3_8b_hipgraph_tp2": {
            "task_info": "data/model/qwen3/q_r_new_model_py.json",
            "smoke_args": "--use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 --warm_up 0 --use_aiter_pa 1 --seq_size_per_block 16 --act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000 --enable_cuda_graph 1 --enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4,5,6,7,8' --tp_size 2 --world_size 2",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_dense_qwen3_8b_ptpc": {
            "task_info": "data/model/qwen3/ptpc_q_r_8b.json",
            "smoke_args": "--quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 --warm_up 0 --use_aiter_pa 1 --seq_size_per_block 16 --act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
    },
    "smoke_rocm_moe": {
        "rocm_moe_qwen3_30b_basic": {
            "task_info": "data/model/qwen3_moe/q_r_30b_amd_py.json",
            "smoke_args": "--quantization FP8_PER_CHANNEL_COMPRESSED --use_asm_pa 1 --act_type BF16 --reserver_runtime_mem_mb 51200 --tp_size 1 --world_size 1 --ep_size 1",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_moe_qwen3_30b_tp2": {
            "task_info": "data/model/qwen3_moe/q_r_30b_amd_py_tp2.json",
            "smoke_args": "--quantization FP8_PER_CHANNEL_COMPRESSED --use_asm_pa 1 --act_type BF16 --reserver_runtime_mem_mb 51200 --tp_size 2 --world_size 2 --ep_size 1",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
    },
    "smoke_rocm_eagle": {
        "rocm_eagle_qwen2_14b": {
            "task_info": "data/model/qwen2_14b/q_r_mtp_rocm.json",
            "smoke_args": "--max_seq_len 16384 --tp_size 1 --use_asm_pa 1 --ft_disable_custom_ar 1 --sp_type eagle --gen_num_per_cycle 4 --warm_up 0 --act_type BF16 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --reserver_runtime_mem_mb 4002",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
    },
    "smoke_rocm_pd": {
        "rocm_pd_qwen3_8b": {
            "task_info": "data/model/qwen3/q_r_new_model_py.json",
            "smoke_args": {
                "prefill": "--test_block_num 10 --warm_up 0 --seq_size_per_block 16 --act_type bf16 --use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 --use_aiter_pa 1 --use_local 1 --role_type PREFILL --world_size 1",
                "decode": "--test_block_num 10 --warm_up 0 --seq_size_per_block 16 --act_type bf16 --use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 --use_aiter_pa 1 --use_local 1 --role_type DECODE --world_size 1",
            },
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
    },
    "smoke_rocm_embedding": {
        "rocm_embedding_qwen3_32b_ptpc_fp8": {
            "task_info": "data/model/qwen3/ptpc_q_r_fp8_py.json",
            "smoke_args": "--reserver_runtime_mem_mb 107813 --use_aiter_pa 1 --seq_size_per_block 16 --fp8_kv_cache 1",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_bert_st": {
            "task_info": "data/model/bert/q_r.json",
            "smoke_args": "--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_roberta_st": {
            "task_info": "data/model/bert/roberta_q_r.json",
            "smoke_args": "--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_roberta_sparse": {
            "task_info": "data/model/bert/sparse_roberta_q_r.json",
            "smoke_args": "--task_type SPARSE_EMBEDDING --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_roberta_colbert": {
            "task_info": "data/model/bert/colbert_roberta_q_r.json",
            "smoke_args": "--task_type COLBERT_EMBEDDING --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_bert_classifier": {
            "task_info": "data/model/bert/bert_classifier_q_r.json",
            "smoke_args": "--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_roberta_reranker": {
            "task_info": "data/model/bert/reranker_q_r.json",
            "smoke_args": "--task_type RERANKER --max_context_batch_size 10 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_roberta_truncate": {
            "task_info": "data/model/bert/reranker_q_r_base.json",
            "smoke_args": "--task_type RERANKER --max_context_batch_size 10 --act_type FP16 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_bge_reranker_trt_fmha": {
            "task_info": "data/model/bert/classifier_q_r.json",
            "smoke_args": "--enable_trt_fmha 0 --enable_open_source_fmha 0 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
        "rocm_embedding_qwen3_32b_ptpc_fp8_cudagraph": {
            "task_info": "data/model/qwen3/ptpc_q_r_fp8_py.json",
            "smoke_args": "--enable_cuda_graph 1 --reserver_runtime_mem_mb 107813 --use_aiter_pa 1 --seq_size_per_block 16 --fp8_kv_cache 1",
            "gpu_type": "MI308X-ROCM7",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "MI308X_ROCM7"],
            "timeout": 600,
        },
    },
    "smoke_cuda_remote_cache": {
        "remote_cache_basic": {
            "task_info": "data/model/qwen25/q_r_l20_remote_cache.json",
            "smoke_args": "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --write_cache_sync 1 --enable_remote_cache true --enable_device_cache 0",
            "envs": ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
        },
        "remote_cache_basic_async": {
            "task_info": "data/model/qwen25/q_r_l20_remote_cache.json",
            "smoke_args": "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0",
            "envs": ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
            "sleep_time_qr": 10,
        },
        "remote_cache_kill": {
            "task_info": "data/model/qwen25/q_r_l20_remote_cache_kill_remote.json",
            "smoke_args": "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0",
            "envs": ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
            "sleep_time_qr": 10,
            "kill_remote": True,
        },
        "remote_cache_tp2": {
            "task_info": "data/model/qwen25/q_r_l20_remote_cache_tpsize_2.json",
            "smoke_args": "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --tp_size 2 --enable_remote_cache true --enable_device_cache 0 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000",
            "envs": ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
            "sleep_time_qr": 20,
        },
        "remote_cache_pd": {
            "task_info": "data/model/qwen25/q_r_l20_remote_cache_pd_sep.json",
            "smoke_args": {
                "prefill": "--warm_up 0  --reuse_cache 1 --role_type PREFILL --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000",
                "decode": "--warm_up 0  --reuse_cache 1 --role_type DECODE --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0 --reco_put_timeout_ms 12000 --reco_get_timeout_ms 12000 --reco_get_broadcast_timeout 15000 --reco_put_broadcast_timeout 15000",
            },
            "envs": ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
            "sleep_time_qr": 20,
        },
        "remote_cache_match_fail": {
            "task_info": "data/model/qwen25/q_r_l20_remote_cache_match_failure.json",
            "smoke_args": "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0",
            "envs": [
                "SEQ_SIZE_PER_BLOCK=8",
                "KVCM_LOG_LEVEL=DEBUG",
                "ENABLE_DEBUG_SERVICE=TRUE",
                "TEST_MATCH_FAILURE=1",
            ],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
            "sleep_time_qr": 10,
        },
        "remote_cache_write_start_fail": {
            "task_info": "data/model/qwen25/q_r_l20_remote_cache_start_and_finish_failure.json",
            "smoke_args": "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0",
            "envs": [
                "SEQ_SIZE_PER_BLOCK=8",
                "KVCM_LOG_LEVEL=DEBUG",
                "ENABLE_DEBUG_SERVICE=TRUE",
                "TEST_START_WRITE_FAILURE=1",
            ],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
            "sleep_time_qr": 10,
        },
        "remote_cache_write_finish_fail": {
            "task_info": "data/model/qwen25/q_r_l20_remote_cache_start_and_finish_failure.json",
            "smoke_args": "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 0",
            "envs": [
                "SEQ_SIZE_PER_BLOCK=8",
                "KVCM_LOG_LEVEL=DEBUG",
                "ENABLE_DEBUG_SERVICE=TRUE",
                "TEST_FINISH_WRITE_FAILURE=1",
            ],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
            "sleep_time_qr": 10,
        },
        "remote_cache_edge": {
            "task_info": "data/model/qwen25/q_r_l20_cache_edge_case_1_remote_cache.json",
            "smoke_args": "--warm_up 0  --reuse_cache 1 --act_type FP16 --seq_size_per_block 4 --enable_remote_cache true --enable_device_cache 0",
            "envs": ["SEQ_SIZE_PER_BLOCK=4", "KVCM_LOG_LEVEL=DEBUG"],
            "gpu_type": "L20",
            "platform": "cuda",
            "markers": ["smoke", "cuda", "L20"],
            "timeout": 600,
            "sleep_time_qr": 10,
        },
    },
}

COMPOSITE_SUITES = {
    "maga_model_smoke_light": [],
    "maga_model_smoke_full": [
        "smoke_h20_mla",
        "smoke_h20_moe",
        "smoke_h20_dense",
        "smoke_h20_next",
        "smoke_h20_eagle",
        "smoke_sm8x_basic",
        "smoke_sm100_dense",
        "smoke_sm100_moe",
        "smoke_rocm_basic",
        "smoke_rocm_dense",
        "smoke_rocm_moe",
        "smoke_rocm_eagle",
        "smoke_rocm_pd",
        "smoke_rocm_embedding",
        "smoke_cuda_remote_cache",
    ],
}


# ---------------------------------------------------------------------------
# Helper functions for pytest integration
# ---------------------------------------------------------------------------


def get_gpu_count(config):
    """Calculate GPU count from smoke_args."""
    smoke_args = config.get("smoke_args", "")
    if isinstance(smoke_args, dict):
        total = 0
        for role_args in smoke_args.values():
            total += _parse_world_size(role_args)
        return total
    return _parse_world_size(smoke_args)


def _parse_world_size(args_str):
    """Parse --tp_size, --dp_size, --pp_size, --world_size from args string."""
    if not args_str:
        return 1
    parts = args_str.split()
    tp = pp = dp = 1
    world_size = None
    i = 0
    while i < len(parts):
        if parts[i] == "--world_size" and i + 1 < len(parts):
            world_size = int(parts[i + 1])
            i += 2
            continue
        if parts[i] == "--tp_size" and i + 1 < len(parts):
            tp = int(parts[i + 1])
            i += 2
            continue
        if parts[i] == "--dp_size" and i + 1 < len(parts):
            dp = int(parts[i + 1])
            i += 2
            continue
        if parts[i] == "--pp_size" and i + 1 < len(parts):
            pp = int(parts[i + 1])
            i += 2
            continue
        i += 1
    return world_size if world_size is not None else tp * pp * dp


def get_all_suites():
    return list(SMOKE_TESTS.keys())


def get_tests_in_suite(suite_name):
    if suite_name in SMOKE_TESTS:
        return list(SMOKE_TESTS[suite_name].keys())
    if suite_name in COMPOSITE_SUITES:
        tests = []
        for s in COMPOSITE_SUITES[suite_name]:
            if s in SMOKE_TESTS:
                tests.extend(SMOKE_TESTS[s].keys())
        return tests
    return []


def get_tests_for_platform(platform):
    result = []
    for suite_name, suite in SMOKE_TESTS.items():
        for name, config in suite.items():
            if config.get("platform") == platform:
                result.append((name, config))
    return result


def build_smoke_params(pytest_module):
    """Build pytest parametrize params from SMOKE_TESTS."""
    params = []
    for suite_name, suite in SMOKE_TESTS.items():
        for test_name, config in suite.items():
            marks = []
            for marker_name in config.get("markers", []):
                marks.append(getattr(pytest_module.mark, marker_name))
            marks.append(pytest_module.mark.manual)

            if suite_name in COMPOSITE_SUITES.get("maga_model_smoke_light", []):
                marks.append(pytest_module.mark.light)

            gpu_type = config.get("gpu_type", "gpu_cuda12")
            gpu_count = get_gpu_count(config)
            marks.append(pytest_module.mark.gpu(type=gpu_type, count=gpu_count))

            params.append(
                pytest_module.param(
                    test_name,
                    config,
                    id=test_name,
                    marks=marks,
                )
            )
    return params
