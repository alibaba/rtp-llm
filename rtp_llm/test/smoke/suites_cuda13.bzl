load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")
load("@arch_config//:arch_select.bzl", "cuda13_x86_test_gpu")

def cuda13_suites():
    # ============================================================================
    # DeepSeek-V4 smoke — minimal production-config coverage
    #
    # Production DSv4 serving always runs decode CUDA graph + PD separation +
    # speculative decode (DSpark or MTP) + prefill CP / decode EP. The feature-off
    # permutations that used to live here (tp1 no-opt, PDFUSION, cp_rr off,
    # CP-overlap off, async-reads off, SWA-gather off, non-PD MTP, mega-MoE SE)
    # gated no shipped code path, so they are gone. What remains is the smallest
    # set that still covers every branch we deploy.
    #
    # Split by executor pool, because the two arches need different CUDA 13
    # configs and different Aone runners:
    #   smoke_cuda13_arm  SM100_ARM_CU13  GB200        --config=cuda13_arm
    #   smoke_cuda13_x86  SM103 pool      SM103        --config=cuda13 (SM 10.3)
    #
    # SM103 carries the Pro checkpoint, which exceeds GB200 memory at EP=4.
    #
    # The x86 cases target the dedicated test pool rather than the shared dev
    # pool: the shared pool is used by other users' work, which left these cases
    # waiting hours for a slot on its one node, while the test pool is dedicated
    # to this pipeline.
    # ============================================================================

    # ARM coverage — V4 production regression and V4.1 full-feature CP2/DP2:
    #   deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm
    #                          CED bounded replay + DSpark + Engram + verify graph
    #                          + prefix reuse, CP2 prefill and DP2 decode (TP=1).
    # Existing V4 coverage:
    #   *_dspark_cprr_async_xgrammar_json
    #                          DSpark speculative decode + CP page-RR ON + CP
    #                          overlap + think mode + Mega MoE SE + PD prefill
    #                          CP2/EP2 + decode DP2/EP2 — closest match to the
    #                          production config. Its fixture already carries
    #                          the json prompt styles (legacy json_format,
    #                          OpenAI response_format json_object / json_schema),
    #                          so the standalone xgrammar case retired without
    #                          porting anything.
    # Coverage intentionally dropped on ARM with the retired cases:
    #   - 65k-context run + sequential in-memory cache-hit assertions
    #     (reuse case; the consolidated case runs 8k context)
    #   - cp_rr=OFF
    #   - the non-SE mega_moe long-generation form
    #   - MTP speculative decode (DSpark remains)
    native.test_suite(
        name = "smoke_cuda13_arm",
        tests = [
            ":deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm",
            # DSpark speculative decode with CP page-RR. This is the blocking
            # regression for publishing both target and draft SWA_KV before decode
            # dispatch; missing draft publication fails with EC_FAILED_LOAD_BUFFER.
            # Both roles pin --moe_strategy mega_moe_se (FP8 shared expert fused
            # into the routed kernel, the only mega-moe form deep_gemm 2.6.1
            # exports), taking over the SE coverage from the retired dedicated
            # mega_moe_se case on this same CP2/DP2 page-RR topology.
            smoke_test(
                name="smoke_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_cprr_async_xgrammar_json_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_async_xgrammar_json_sm100_arm.json",
                sleep_time_qr=10,
                smoke_args={
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 69632 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 1 --enable_fp32_lm_head 0",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --decode_capture_config '1,2,4,8,16' --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 30000 --reserver_runtime_mem_mb 10240 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2 --think_mode 1 --enable_fp32_lm_head 0",
                },
                envs={
                    "prefill": [
                        "ENABLE_DSPARK=1",
                        "SP_TYPE=dspark",
                        "SP_MODEL_TYPE=deepseek_v4_dspark",
                        "SP_CHECKPOINT_PATH=/mnt/nas1/hf/DeepSeek-V4-Flash-DSpark",
                        "SP_ACT_TYPE=bf16",
                        "GEN_NUM_PER_CIRCLE=3",
                        "MODEL_EVAL_VARIANT=fp8_kv_cache",
                        "DSV4_BF16_VLLM=0",
                        "DSV4_COMPRESSOR_FAST=1",
                        "DSV4_COMPRESSOR_METADATA_TRITON=0",
                        "DSV4_MHC_PRE_GEMM_BACKEND=tilelang_single",
                        "DSV4_FIXED_POOL_USE_MEMORY=0",
                        "DSV4_TRAP_INVALID_KV_ACCESS=0",
                        "DSV4_PREFILL_CP_OVERLAP=1",
                        "DSV4_CHUNK_TOKENS=12288",
                        "ENABLE_GPU_PREFIX_TREE=1",
                        "ENABLE_DSV4_STATE_BLOCK_INDEPENDENT_EVICTION=0",
                        "ENABLE_LEGACY_MEMORY_CONNECTOR_FALLBACK=0",
                        "CP_FORCE_SINGLE_PREFILL=0",
                        "PREFILL_CP_KV_CACHE_SHARDED=1",
                        "RTP_LLM_DEVICE_INPUT=1",
                        "ENABLE_LAYER_MICRO_BATCH=0",
                        "LINEAR_STEP=1",
                        "DG_JIT_CPP_STANDARD=20",
                        "DG_MEGA_MOE_NVLINK_BARRIER_TIMEOUT_SECS=300",
                        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:1024",
                        "FRONTEND_SERVER_COUNT=1",
                        "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS=500",
                        "THINK_MODE=1",
                        "THINK_START_TAG=<think>",
                        "THINK_END_TAG=</think>",
                    ],
                    "decode": [
                        "ENABLE_DSPARK=1",
                        "SP_TYPE=dspark",
                        "SP_MODEL_TYPE=deepseek_v4_dspark",
                        "SP_CHECKPOINT_PATH=/mnt/nas1/hf/DeepSeek-V4-Flash-DSpark",
                        "SP_ACT_TYPE=bf16",
                        "GEN_NUM_PER_CIRCLE=3",
                        "MODEL_EVAL_VARIANT=fp8_kv_cache",
                        "DSV4_BF16_VLLM=0",
                        "DSV4_COMPRESSOR_FAST=1",
                        "DSV4_MHC_PRE_GEMM_BACKEND=tilelang_single",
                        "OMP_NUM_THREADS=8",
                        "RTP_LLM_STREAM_ASYNC=1",
                        "RTP_LLM_DROP_BROAD_SYNC=1",
                        "RTP_LLM_DEVICE_INPUT=1",
                        "RTP_LLM_MTP_ASYNC_PREPARE=0",
                        "ENABLE_LAYER_MICRO_BATCH=0",
                        "LINEAR_STEP=1",
                        "DG_JIT_CPP_STANDARD=20",
                        "DG_JIT_CACHE_DIR=__TEST_TMPDIR__/deep_gemm_pd_cache",
                        "THINK_MODE=1",
                        "THINK_START_TAG=<think>",
                        "THINK_END_TAG=</think>",
                    ],
                },
                gpu_type=["SM100_ARM_CU13"],
            ),
        ],
        tags = ["manual"],
    )

    # X86 coverage: the Pro checkpoint requires the larger SM103 memory.
    native.test_suite(
        name = "smoke_cuda13_x86",
        tests = [
            # Real CRC corruption checks dump, zero-reuse fallback and detached-node reclamation.
            ":crc_failure_smoke",
            # CRC write/read regression with real HOST reuse; DEVICE/DISK/remote
            # reuse are disabled so the second query must reload HOST blocks.
            # A final multi-token request also exercises DSpark decode in this
            # same CP page-RR, 2P + 2D server, without depending on grammar pins.
            smoke_test(
                name = "smoke_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_crc_memory_sm103",
                task_info = "data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_crc_memory_sm103.json",
                smoke_args = {
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --enable_disk_cache 0 --enable_remote_cache 0 --memory_cache_size_mb 1024 --memory_cache_sync_timeout_ms 120000 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 0 --enable_fp32_lm_head 0",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 0 --enable_fp32_lm_head 0 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                envs = {
                    "prefill": [
                        "DSV4_USE_FRAMEWORK_KV=1",
                        "DSV4_PREFILL_CP_OVERLAP=1",
                        "DG_JIT_CPP_STANDARD=20",
                        "LOG_LEVEL=DEBUG",
                    ],
                    "decode": ["DSV4_USE_FRAMEWORK_KV=1", "DG_JIT_CPP_STANDARD=20", "LOG_LEVEL=DEBUG"],
                },
                gpu_type = [cuda13_x86_test_gpu()],
                sleep_time_qr = 10,
            ),

            # Long HOST prefixes exercise large CRC transfer batches,
            # alongside the short-prefix cases.
            smoke_test(
                name = "smoke_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_large_batch_crc_memory_sm103",
                task_info = "data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_dp2ep2_dspark_cprr_large_batch_crc_memory_sm103.json",
                smoke_args = {
                    "prefill": "--memory_cache_max_descriptors_per_transfer_batch 32 --load_method fastsafetensors --max_seq_len 12288 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --enable_disk_cache 0 --enable_remote_cache 0 --memory_cache_size_mb 1024 --memory_cache_sync_timeout_ms 120000 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 0 --enable_fp32_lm_head 0",
                    "decode": "--memory_cache_max_descriptors_per_transfer_batch 32 --load_method fastsafetensors --max_seq_len 12288 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 0 --enable_fp32_lm_head 0 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                envs = {
                    "prefill": [
                        "DSV4_USE_FRAMEWORK_KV=1",
                        "DSV4_PREFILL_CP_OVERLAP=1",
                        "DG_JIT_CPP_STANDARD=20",
                        "LOG_LEVEL=DEBUG",
                    ],
                    "decode": ["DSV4_USE_FRAMEWORK_KV=1", "DG_JIT_CPP_STANDARD=20", "LOG_LEVEL=DEBUG"],
                },
                gpu_type = [cuda13_x86_test_gpu()],
                sleep_time_qr = 10,
            ),

            # Only DeepSeek-V4-Pro case. Deliberately plain — no CUDA graph, no PD
            # separation, no speculative decode, no page-RR. The advanced feature
            # matrix is covered on Flash above; this exists so a Pro-specific
            # regression (weight layout, expert count, rope/scaling config) cannot
            # ship unnoticed.
            #
            # Single-role CP=4 + EP=4 all-gather prefill topology, 4 GPUs. Runs on
            # SM103 rather than GB200: Pro is ~865GB, so EP=4 needs ~216GB/rank,
            # over GB200's ~186GB but inside SM103's ~288GB.
            smoke_test(
                name="v4_pro_cp4_ep4_basic_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_pro_cp4_sm100_arm.json",
                smoke_args="--load_method scratch --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 4 --dp_size 1 --ep_size 4 --moe_strategy mega_moe_se --world_size 4 --seq_size_per_block 256 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --concurrency_limit 1 --max_context_batch_size 1 --reserver_runtime_mem_mb 20480",
                envs=["DG_JIT_CPP_STANDARD=20"],
                gpu_type=[cuda13_x86_test_gpu()],
            ),
        ],
        tags = ["manual"],
    )
