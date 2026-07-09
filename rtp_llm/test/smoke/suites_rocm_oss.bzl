load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def rocm_oss_suites():
    # ROCm (AMD MI308X) — INDEPENDENT suite tree, all cases prefixed with `rocm_`
    # ============================================================================

    # ROCm basic framework features
    native.test_suite(
        name = "smoke_rocm_basic",
        tests = [
            smoke_test(
                name="rocm_basic_cache_reuse",
                task_info="data/model/qwen2/q_r_reuse.json",
                smoke_args="--reuse_cache 1 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_basic_batch_cache_reuse",
                task_info="data/model/qwen3/q_r_308x_batch_cache.json",
                smoke_args="--reuse_cache 1 --enable_cuda_graph 1 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_basic_beam_search_tp2",
                task_info="data/model/qwen25/bs_q_r_mi308x.json",
                smoke_args="--tp_size 2 --warm_up 0 --seq_size_per_block 16 --use_asm_pa 0 --use_aiter_pa 1 --disable_flash_infer 1 --act_type BF16",
                gpu_type=["MI308X-ROCM7"],
            ),
        ],
    )


    # ROCm Dense (Qwen3 dense — covered cases removed; ptpc 32B kept pending smaller-model swap)
    native.test_suite(
        name = "smoke_rocm_dense",
        tests = [
            smoke_test(
                name="rocm_dense_qwen3_8b_hipgraph_tp2",
                task_info="data/model/qwen3/q_r_new_model_py.json",
                smoke_args="--use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 --warm_up 0 --use_aiter_pa 1 --seq_size_per_block 16 --act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000 --enable_cuda_graph 1 --enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4,5,6,7,8' --tp_size 2 --world_size 2",
                gpu_type=["MI308X-ROCM7"]
            ),
            # Simplified from Qwen3-32B-FP8-Dynamic → Qwen3-8B; result placeholder, needs rewrite_smoke regen on MI308X
            smoke_test(
                name="rocm_dense_qwen3_8b_ptpc",
                task_info="data/model/qwen3/ptpc_q_r_8b.json",
                smoke_args="--quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 --warm_up 0 --use_aiter_pa 1 --seq_size_per_block 16 --act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_dense_qwen3_8b_ptpc_fp8kv_asm_pa",
                task_info="data/model/qwen3/ptpc_q_r_8b.json",
                smoke_args="--quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --use_asm_pa 1 --fp8_kv_cache 1 --enable_cuda_graph 1 --warm_up 1 --act_type BF16 --reserver_runtime_mem_mb 70000 --test_block_num 1000",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_dense_qwen3_8b_ptpc_fp8kv_no_asm_pa",
                task_info="data/model/qwen3/ptpc_q_r_8b.json",
                smoke_args="--quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --use_asm_pa 0 --fp8_kv_cache 1 --enable_cuda_graph 1 --warm_up 1 --act_type BF16 --reserver_runtime_mem_mb 70000 --test_block_num 1000",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_dense_qwen3_8b_ptpc_no_asm_pa",
                task_info="data/model/qwen3/ptpc_q_r_8b.json",
                smoke_args="--quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --use_asm_pa 0 --disable_flash_infer 1 --warm_up 0 --use_aiter_pa 1 --seq_size_per_block 16 --act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000",
                gpu_type=["MI308X-ROCM7"],
            ),
        ],
    )


    # ROCm MoE (Qwen3-30B MoE)
    native.test_suite(
        name = "smoke_rocm_moe",
        tests = [
            smoke_test(
                name="rocm_moe_qwen3_30b_basic",
                task_info="data/model/qwen3_moe/q_r_30b_amd_py.json",
                smoke_args="--quantization FP8_PER_CHANNEL_COMPRESSED --use_asm_pa 1 --act_type BF16 --reserver_runtime_mem_mb 51200 --tp_size 1 --world_size 1 --ep_size 1",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_moe_qwen3_30b_tp2",
                task_info="data/model/qwen3_moe/q_r_30b_amd_py_tp2.json",
                smoke_args="--quantization FP8_PER_CHANNEL_COMPRESSED --use_asm_pa 1 --act_type BF16 --reserver_runtime_mem_mb 51200 --tp_size 2 --world_size 2 --ep_size 1",
                gpu_type=["MI308X-ROCM7"],
            ),
        ],
    )

    # ROCm Qwen3.5 (MI308X): compact online-shape coverage.
    native.test_suite(
        name = "smoke_rocm_qwen35",
        tests = [
            smoke_test(
                name="rocm_qwen35_bf16_reuse",
                task_info="data/model/qwen35/qwen35_bf16_rocm_reuse.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 1024 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 409600 --tp_size 1 --world_size 1 --use_asm_pa 1 --use_aiter_pa 1 --use_triton_pa 1 --reserver_runtime_mem_mb 40480 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_qwen35_fp8_cg_reuse",
                task_info="data/model/qwen35/qwen35_fp8_rocm_cg_reuse.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 1024 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 409600 --tp_size 1 --world_size 1 --use_asm_pa 1 --use_aiter_pa 1 --use_triton_pa 1 --reserver_runtime_mem_mb 40480 --quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --write_cache_sync 1 --enable_cuda_graph 1 --enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4'",
                gpu_type=["MI308X-ROCM7"],
                data=native.glob(['data/model/qwen_vl/*.jpeg']),
            ),
            smoke_test(
                name="rocm_qwen35_fp8_tp2_cg_reuse",
                task_info="data/model/qwen35/qwen35_fp8_rocm_tp2_cg_reuse.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 1024 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 409600 --tp_size 2 --world_size 2 --ep_size 1 --use_asm_pa 1 --use_aiter_pa 1 --use_triton_pa 1 --reserver_runtime_mem_mb 40480 --quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --reuse_cache 1 --enable_cuda_graph 1 --enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4'",
                gpu_type=["MI308X-ROCM7"],
            ),
            # Qwen3.5-27B TP2 packs 422604 tokens: the QKV source offset reaches
            # 2163727360 and the FLA chunk-state offset reaches 2597191680.
            smoke_test(
                name="rocm_qwen35_27b_fp8_pd_tp2x2_cg_long_prefill",
                task_info="data/model/qwen35/qwen35_27b_fp8_pd_tp2x2_cg_long_prefill.json",
                smoke_args={
                    "prefill": "--load_cache_timeout_ms 900000 --warm_up 0 --act_type BF16 --seq_size_per_block 1024 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 131072 --max_context_batch_size 4 --max_batch_tokens_size 524288 --tp_size 2 --world_size 2 --ep_size 1 --use_asm_pa 1 --use_aiter_pa 1 --use_triton_pa 1 --reserver_runtime_mem_mb 40480 --quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --reuse_cache 0 --concurrency_limit 4 --cache_store_rdma_mode 0 --use_local 1 --role_type PREFILL",
                    "decode": "--load_cache_timeout_ms 900000 --warm_up 0 --act_type BF16 --seq_size_per_block 1024 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 131072 --max_context_batch_size 4 --max_batch_tokens_size 524288 --tp_size 2 --world_size 2 --ep_size 1 --use_asm_pa 1 --use_aiter_pa 1 --use_triton_pa 1 --reserver_runtime_mem_mb 40480 --quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 --reuse_cache 0 --concurrency_limit 4 --cache_store_rdma_mode 0 --use_local 1 --enable_cuda_graph 1 --enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4' --role_type DECODE",
                },
                gpu_type=["MI308X-ROCM7"],
            ),
        ],
    )


    # ROCm Eagle (Qwen2-14B + draft)
    native.test_suite(
        name = "smoke_rocm_eagle",
        tests = [
            smoke_test(
                name="rocm_eagle_qwen2_14b",
                task_info="data/model/qwen2_14b/q_r_mtp_rocm.json",
                smoke_args="--max_seq_len 16384 --tp_size 1 --use_asm_pa 1 --ft_disable_custom_ar 1 --sp_type eagle --gen_num_per_cycle 4 --warm_up 0 --act_type BF16 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --reserver_runtime_mem_mb 4002",
                gpu_type=["MI308X-ROCM7"]
            ),
        ],
    )


    # ROCm PD seperation
    native.test_suite(
        name = "smoke_rocm_pd",
        tests = [
            smoke_test(
                name="rocm_pd_qwen3_8b",
                task_info="data/model/qwen3/q_r_new_model_py.json",
                smoke_args= {
                    "prefill": "--test_block_num 10 --warm_up 0 --seq_size_per_block 16 --act_type bf16 --use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 --use_aiter_pa 1 --use_local 1 --role_type PREFILL --world_size 1",
                    "decode": "--test_block_num 10 --warm_up 0 --seq_size_per_block 16 --act_type bf16 --use_swizzleA 1 --use_asm_pa 1 --disable_flash_infer 1 --use_aiter_pa 1 --use_local 1 --role_type DECODE --world_size 1"
                },
                gpu_type=["MI308X-ROCM7"]
            ),
            smoke_test(
                name="rocm_pd_qwen35_bf16_tp2_cg",
                task_info="data/model/qwen35/qwen35_bf16_rocm.json",
                smoke_args= {
                    "prefill": "--load_cache_timeout_ms 120000 --warm_up 0 --act_type BF16 --seq_size_per_block 1024 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 409600 --tp_size 2 --world_size 2 --use_asm_pa 1 --use_aiter_pa 1 --use_triton_pa 1 --reserver_runtime_mem_mb 40480 --cache_store_rdma_mode 0 --use_local 1 --role_type PREFILL",
                    "decode": "--load_cache_timeout_ms 120000 --warm_up 0 --act_type BF16 --seq_size_per_block 1024 --kernel_seq_size_per_block 16 --test_block_num 512 --max_seq_len 409600 --tp_size 2 --world_size 2 --use_asm_pa 1 --use_aiter_pa 1 --use_triton_pa 1 --reserver_runtime_mem_mb 40480 --cache_store_rdma_mode 0 --use_local 1 --enable_cuda_graph 1 --enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4' --role_type DECODE"
                },
                gpu_type=["MI308X-ROCM7"]
            ),
        ],
    )


    # ROCm Embedding (BERT/RoBERTa)
    native.test_suite(
        name = "smoke_rocm_embedding",
        tests = [
            smoke_test(
                name="rocm_embedding_qwen3_32b_ptpc_fp8",
                task_info="data/model/qwen3/ptpc_q_r_fp8_py.json",
                gpu_type=["MI308X-ROCM7"],
                smoke_args="--reserver_runtime_mem_mb 107813 --use_aiter_pa 1 --seq_size_per_block 16 --fp8_kv_cache 1",
            ),
            smoke_test(
                name="rocm_embedding_bert_st",
                task_info="data/model/bert/q_r.json",
                smoke_args="--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_embedding_roberta_st",
                task_info="data/model/bert/roberta_q_r.json",
                smoke_args="--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_embedding_roberta_sparse",
                task_info="data/model/bert/sparse_roberta_q_r.json",
                smoke_args="--task_type SPARSE_EMBEDDING --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
                gpu_type=["MI308X-ROCM7"],
            ),
            # TODO(yanquan): ~5% flaky GPU memory access fault on MI308X, likely async memory race
            # in aiter CK attention kernel during forward pass. See build_logs/rocm_colbert_crash_investigation.md
            # smoke_test(
            #     name="rocm_embedding_roberta_colbert",
            #     task_info="data/model/bert/colbert_roberta_q_r.json",
            #     smoke_args="--task_type COLBERT_EMBEDDING --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
            #     gpu_type=["MI308X-ROCM7"],
            # ),
            smoke_test(
                name="rocm_embedding_bert_classifier",
                task_info="data/model/bert/bert_classifier_q_r.json",
                smoke_args="--seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_embedding_roberta_reranker",
                task_info="data/model/bert/reranker_q_r.json",
                smoke_args="--task_type RERANKER --max_context_batch_size 10 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_embedding_roberta_truncate",
                task_info="data/model/bert/reranker_q_r_base.json",
                smoke_args="--task_type RERANKER --max_context_batch_size 10 --act_type FP16 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_embedding_bge_reranker_trt_fmha",
                task_info="data/model/bert/classifier_q_r.json",
                smoke_args="--enable_trt_fmha 0 --enable_open_source_fmha 0 --seq_size_per_block 16 --use_aiter_pa 1 --use_asm_pa 1 --act_type FP16",
                gpu_type=["MI308X-ROCM7"],
            ),
            smoke_test(
                name="rocm_embedding_qwen3_32b_ptpc_fp8_cudagraph",
                task_info="data/model/qwen3/ptpc_q_r_fp8_py.json",
                gpu_type=["MI308X-ROCM7"],
                smoke_args="--enable_cuda_graph 1 --reserver_runtime_mem_mb 107813 --use_aiter_pa 1 --seq_size_per_block 16 --fp8_kv_cache 1",
            ),
        ],
    )

    # ROCm VL (Qwen3-VL dense)
    native.test_suite(
        name = "smoke_rocm_vl",
        tests = [
            smoke_test(
                name="rocm_qwen3_vl_cg",
                task_info="data/model/qwen_vl/q_r_3.json",
                smoke_args="--act_type BF16 --reserver_runtime_mem_mb 51200 --tp_size 1 --world_size 1 --enable_cuda_graph 1 --enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4'",
                gpu_type=["MI308X-ROCM7"],
            ),
        ],
    )
