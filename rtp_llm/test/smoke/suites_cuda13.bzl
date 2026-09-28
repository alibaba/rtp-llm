load("//rtp_llm/test/smoke:defs.bzl", "SMOKE_FRAMEWORK_DEPS", "custom_smoke_test", "smoke_test")

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
    #   smoke_cuda13_x86  L20D_TEST       L20D / B300  --config=cuda13 (SM 10.3)
    #
    # B300 capacity is the scarce one — a single dedicated node — so x86 carries
    # the two capacity-sensitive cases that GB200 cannot host, plus the tier-isolated
    # BlockTreeCache and FlexLB cache-affinity cases that are recorded for B300.
    #
    # The x86 cases target L20D_TEST rather than L20D_DEV: L20D_DEV is shared with
    # other users' work, which left these cases waiting hours for a slot on its one
    # node, while L20D_TEST is dedicated to this pipeline.
    # ============================================================================

    # ARM coverage:
    #   *_reuse_memory_cache   PD sep + prefill CP2 + decode DP2/EP2 + decode CUDA
    #                          graph + in-memory prefix reuse + routed-only Mega MoE.
    #                          Also the cp_rr=OFF half of the required CP
    #                          page-RR coverage (no --prefill_cp_kv_cache_sharded)
    #                          — keep it that way.
    #   *_xgrammar_json        legacy json_format + OpenAI response_format
    #                          json_object / json_schema over PD
    #   *_mega_moe_se          MTP + CP page-RR + Mega MoE with the FP8 shared
    #                          expert fused in-kernel, under long generation.
    #   *_dspark_cprr_async_xgrammar_json
    #                          DSpark speculative decode + CP page-RR + CP overlap
    #                          + think mode + Mega MoE — closest match to the
    #                          production config.
    native.test_suite(
        name = "smoke_cuda13_arm",
        tests = [
            ###
            # Comprehensive PD reuse-cache smoke: prefill CP=2×EP=2 (2 GPUs) +
            # decode DP=2×EP=2 (2 GPUs) = 4 GPUs total.
            #
            # What this exercises in one server start:
            #   - fastsafetensors weight loader (new-path V4Weight descriptors)
            #   - KV-prefix reuse cache (--reuse_cache 1)
            #   - In-memory cache layer on top (--enable_memory_cache 1)
            #   - Context Parallel prefill (CP=2, ALL_GATHER rotate)
            #   - Routed-only Mega MoE on both sides (--moe_strategy mega_moe)
            #   - Decode CUDA graph (--enable_cuda_graph 1, BS captures 1/2/4/8)
            #   - max_seq_len=65600 — handles 64k-token prefill context
            #
            # 5 sequential requests spanning short / medium / long contexts:
            #   Q0: "What is the capital of France?" (11 tok, cold)
            #   Q1: same short prompt (cache hit — verifies short-prefix reuse)
            #   Q2: 15k AI-history doc (cold — fills the KV cache)
            #   Q3: same 15k prefix, max_tokens=10 (memory cache hit)
            #   Q4: 64k summary prompt (cold — exercises CP=2 all-gather at max context)
            #
            # GPU allocation: PD runner slices (0,1) → prefill, (2,3) → decode.
            # Decode capture includes 32 because this topology's startup warmup
            # reaches batch size 32.
            #
            # Cache writes publish asynchronously. Wait between sequential queries
            # so reuse assertions observe completed publication. This host-cache
            # case does not cover DISK; disk transfer and PD handoff deadlines need
            # separate coverage.
            #
            # Q2/Q4 goldens record cached_tokens even though both prompts are cold:
            # in PD the decode only recomputes the trailing partial block and takes
            # the block-aligned prefix from the prefill over the cache store, and
            # that transferred prefix is what reuse accounting reports. Q3 is the
            # real reuse assertion and compares the memory-cache hit instead.
            smoke_test(
                name="v4_flash_pd_cp2ep2_dp2ep2_reuse_memory_cache_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_reuse_cache_sm100_arm.json",
                sleep_time_qr=10,
                smoke_args={
                    "prefill": "--load_method fastsafetensors --max_seq_len 65600 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --moe_strategy mega_moe --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1",
                    "decode": "--load_method fastsafetensors --max_seq_len 65600 --enable_cuda_graph 1 --decode_capture_config '1,2,4,8,32' --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --moe_strategy mega_moe --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --use_deepep_moe 1 --use_deepep_low_latency 1 --cp_rotate_method PREFILL_CP --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1",
                },
                gpu_type=["SM100_ARM_CU13"],
            ),
            # Short xgrammar regression fixture for DeepSeek-V4-Flash PD 1P1D.
            # Covers legacy json_format, OpenAI response_format=json_object, and
            # response_format=json_schema without paying the 1M-context cost.
            # Both roles are single-card (tp1/ep1/dp1/world1), which leaves only
            # ~24.8 GiB of HBM after weights, so the runtime reservation has to stay
            # well under that or MemoryEvaluationHelper rejects it outright.
            #
            # The fixture only asserts what the comparer can actually assert: the
            # dash_sc_grpc queries and the "expect a 400" negative queries are gone,
            # because a rejected request surfaces as VISIT_FAILED / OTHERS rather
            # than as a matched expectation. Free-form generations keep
            # json_content + required_json_keys and drop expected_json — the schema
            # is the invariant, the prose behind it is not. Under
            # response_format=json_object the model also picks its own key names
            # ("condition" vs "weather"), so those queries assert only that the
            # content parses as a JSON object; key names are asserted where a
            # json_schema pins them.
            smoke_test(
                name="v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_flash_pd_cp4_tp1ep1dp1_xgrammar_json_sm100_arm.json",
                smoke_args={
                    "prefill": "--load_method scratch --force_cpu_load_weights 1 --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 1 --ep_size 1 --world_size 1 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 4096 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --warm_up 1 --reserver_runtime_mem_mb 10240",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 1 --dp_size 1 --ep_size 1 --world_size 1 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 4096 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 --concurrency_limit 4 --reserver_runtime_mem_mb 10240",
                },
                envs={
                    "prefill": [
                        "RTP_LLM_STREAM_ASYNC=1",
                        "RTP_LLM_DROP_BROAD_SYNC=1",
                        "RTP_LLM_DEVICE_INPUT=1",
                    ],
                    "decode": [
                        "RTP_LLM_STREAM_ASYNC=1",
                        "RTP_LLM_DROP_BROAD_SYNC=1",
                        "RTP_LLM_DEVICE_INPUT=1",
                    ],
                },
                gpu_type=["SM100_ARM_CU13"],
            ),
            # Dedicated coverage for the routed-only Mega MoE strategy.
            smoke_test(
                name="v4_flash_mega_moe_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_flash_mega_moe_sm100.json",
                sleep_time_qr=10,
                smoke_args={
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 8192 --use_deepep_moe 0 --use_deepep_low_latency 0 --moe_strategy mega_moe --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --use_deepep_moe 0 --use_deepep_low_latency 0 --moe_strategy mega_moe --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                gpu_type=["SM100_ARM_CU13"],
            ),
            # Mega MoE with the FP8 shared expert fused into the routed kernel, on
            # the same Flash CP2/DP2 page-RR + MTP topology as the x86 logits case
            # (2 prefill + 2 decode GPUs = GB200's 4 cards). This is the SE path,
            # not the old standalone fused path: deep_gemm 2.6.1 exports only
            # ``fp8_fp4_mega_moe`` with optional shared-expert arguments, and
            # dropped the ``*_mega_moe_fused`` family entirely, so
            # DSV4_USE_MEGA_MOE_FUSED=1 can no longer pass strict strategy
            # selection on any platform.
            smoke_test(
                name="v4_flash_mega_moe_se_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_flash_mega_moe_se_sm100.json",
                sleep_time_qr=10,
                smoke_args={
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 8192 --use_deepep_moe 0 --use_deepep_low_latency 0 --moe_strategy mega_moe_se --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --use_deepep_moe 0 --use_deepep_low_latency 0 --moe_strategy mega_moe_se --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                gpu_type=["SM100_ARM_CU13"],
            ),
            # DSpark speculative decode with CP page-RR. This is the blocking
            # regression for publishing both target and draft SWA_KV before decode
            # dispatch; missing draft publication fails with EC_FAILED_LOAD_BUFFER.
            smoke_test(
                name="smoke_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_cprr_async_xgrammar_json_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_async_xgrammar_json_sm100_arm.json",
                sleep_time_qr=10,
                smoke_args={
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 8192 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 69632 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --think_mode 1 --enable_fp32_lm_head 0",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --decode_capture_config '1,2,4,8,16' --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 30000 --reserver_runtime_mem_mb 10240 --fp8_kv_cache 1 --sp_type dspark --gen_num_per_cycle 3 --sp_model_type deepseek_v4_dspark --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash-DSpark --sp_act_type bf16 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2 --think_mode 1 --enable_fp32_lm_head 0",
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

    # X86 coverage — capacity-sensitive cases plus tier-isolated page-RR golden coverage:
    #   *_block_tree_device/only_memory/
    #   *_only_disk_sm100                  Two-query DEVICE/HOST/DISK-only P/D
    #                                      handoff with MTP, logits, and page-RR.
    #   *_1m                               1M-token prefill: chunked Mega MoE buffer,
    #                                      indexer chunked score, int64 row indexing.
    #                                      Block counts and reserved memory are tuned
    #                                      against the B300 node it was recorded on.
    #   v4_pro_cp4_ep4_basic               DeepSeek-V4-Pro, all advanced features
    #                                      off — Pro checkpoint regression only.
    #                                      ~216GB/rank at EP=4, over GB200's ~186GB.
    #
    # CP page-RR (--prefill_cp_kv_cache_sharded) is covered both ways:
    #   ON   *_mega_moe_se and *_dspark_cprr_async_xgrammar_json on ARM, plus
    #        the three tier-isolated x86 BlockTree cases
    #   OFF  *_reuse_memory_cache
    # The x86 *_1m case runs cp_rr off as well, but ARM already gates that half.
    native.test_suite(
        name = "smoke_cuda13_x86",
        tests = [
            # Tier-isolated two-query P/D handoff regressions. These preserve the
            # legacy SM100 x86 golden payloads while using the existing shared cloud-disk model.
            smoke_test(
                name = "smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_block_tree_device_only_sm100",
                task_info = "data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_block_tree_device_only_sm100_arm.json",
                smoke_args = {
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 1024 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0 --test_block_num 34 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 1024 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                envs = {
                    "prefill": [
                        "DSV4_USE_FRAMEWORK_KV=1",
                        "LOG_LEVEL=DEBUG",
                    ],
                    "decode": ["DSV4_USE_FRAMEWORK_KV=1"],
                },
                gpu_type = ["L20D_TEST"],
                sleep_time_qr = 1,
            ),

            # HOST/DISK use 23 physical working blocks: block 0 leaves 22 usable,
            # covering the 8188-token warmup's 17 planned + 5 reserved blocks.
            smoke_test(
                name = "smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_block_tree_only_memory_sm100",
                task_info = "data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_block_tree_only_memory_sm100_arm.json",
                smoke_args = {
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --enable_disk_cache 0 --enable_remote_cache 0 --memory_cache_size_mb 32 --memory_cache_sync_timeout_ms 120000 --test_block_num 23 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                envs = {
                    "prefill": [
                        "DSV4_USE_FRAMEWORK_KV=1",
                        "LOG_LEVEL=DEBUG",
                    ],
                    "decode": ["DSV4_USE_FRAMEWORK_KV=1"],
                },
                gpu_type = ["L20D_TEST"],
                sleep_time_qr = 1,
            ),

            smoke_test(
                name = "smoke_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_block_tree_only_disk_sm100",
                task_info = "data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_dp2ep2_mtp_page_rr_logits_block_tree_only_disk_sm100_arm.json",
                smoke_args = {
                    "prefill": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 0 --enable_disk_cache 1 --enable_remote_cache 0 --disk_cache_staging_block_count 2 --test_block_num 23 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --reserver_runtime_mem_mb 65536 --max_context_batch_size 1 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16",
                    "decode": "--load_method fastsafetensors --max_seq_len 8192 --enable_cuda_graph 1 --act_type BF16 --tp_size 1 --dp_size 2 --ep_size 2 --world_size 2 --seq_size_per_block 256 --kernel_seq_size_per_block 128 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0 --use_deepep_moe 1 --use_deepep_low_latency 1 --load_cache_timeout_ms 120000 --reserver_runtime_mem_mb 49152 --fp8_kv_cache 1 --sp_type mtp --gen_num_per_cycle 3 --sp_model_type deepseek_v4_mtp --sp_checkpoint_path /mnt/nas1/hf/DeepSeek-V4-Flash --sp_act_type bf16 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2",
                },
                envs = {
                    "prefill": [
                        "DSV4_USE_FRAMEWORK_KV=1",
                        "LOG_LEVEL=DEBUG",
                        "DISK_CACHE_PATHS=__TEST_TMPDIR__/disk_kv_only_disk_prefill0,__TEST_TMPDIR__/disk_kv_only_disk_prefill1",
                        "DISK_CACHE_SIZE_MB=64",
                        "DISK_CACHE_BUFFERED_IO=1",
                        "DISK_CACHE_SYNC_TIMEOUT_MS=120000",
                    ],
                    "decode": ["DSV4_USE_FRAMEWORK_KV=1"],
                },
                gpu_type = ["L20D_TEST"],
                sleep_time_qr = 2,
            ),
            # CP=2/EP=2 prefill + TP=1/EP=1/DP=1 single-card decode PD smoke
            # targeting 1M context. Single ~1.09M-token LongBench-V2 q62 query
            # (max_tokens=8) exercises the long-context prefill
            # (cp_rotate_method=ALL_GATHER) and a minimal decode topology — the
            # latter relies on the chunked Mega MoE buffer plus indexer chunked
            # score in this branch to keep decode within a single GPU's HBM.
            #
            # CP=2 rather than CP=4: at CP=4 the case wanted 4+1=5 GPUs, which on
            # an 8-GPU B300 cannot pack alongside the 4-GPU v4_pro case below, so
            # the two serialized and the job outgrew its timeout. At CP=2 it needs
            # 2+1=3 and the pair fits one node. The roles cannot share GPUs
            # (SMOKE_PD_SHARE_GPU=1): decode is world_size=1 and holds the whole
            # ~160GB of weights, which does not co-reside with a prefill rank.
            #
            # Halving the CP degree doubles each prefill rank's share of the
            # device blocks (~2.1K of the 6000 in test_block_num, so still far
            # inside it). ``memory_cache_size_mb`` is per rank and buys host-cache
            # blocks of the same ~5.94MB as the device ones, so the original 393216
            # asked for 69432 blocks == a 412GB host pool on each of the two
            # prefill ranks and the worker's memory cgroup OOM-killed them mid
            # startup. 49152 gives ~8.2K blocks, twice the ~4.2K-block payload each
            # rank caches in full (it caches the whole payload rather than a CP
            # shard, so the requirement is independent of the CP degree).
            #
            # The run also exercises the int64 row-indexing fix for long-context
            # Triton kernels (combine_topk_swa) and the per_token_group_quant_8bit_v2
            # int64 offset fix landed on this branch — both regress as
            # ``CUDA_ERROR_ILLEGAL_ADDRESS`` at L3 HCA without the fix.
            # The golden checks usage and aux_info PD fields, not exact text:
            # the 8-token continuation is not bit-exact on this path.
            # The ~1.075M-token request consumes about 4201 logical
            # 256-token blocks, so 6000 prefill blocks leave roughly 1800 blocks of
            # GPU headroom for the D2H cache-publication staging buffer. Reserving
            # 12000 exhausted that headroom: execStagedMemoryCopy failed to allocate
            # device staging, no cache entry was published, and decode eventually
            # hit LOAD_CACHE_TIMEOUT.
            #
            # ``load_cache_timeout_ms`` stays at the default order of magnitude on
            # purpose: publishing this ~24GB of KV cache batches into one pinned
            # staging buffer per submission, so anything slower than 30s means the
            # producer regressed rather than that the payload is simply large.
            smoke_test(
                name="v4_flash_pd_cp2ep2_tp1ep1dp1_1m_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_flash_pd_cp2ep2_tp1ep1dp1_1m_sm100_arm.json",
                smoke_args={
                    "prefill": "--load_method scratch --force_cpu_load_weights 1 --max_seq_len 1100000 --enable_cuda_graph 0 --act_type BF16 --tp_size 2 --ep_size 2 --moe_strategy mega_moe_se --world_size 2 --seq_size_per_block 256 --test_block_num 6000 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 49152 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --warm_up 1 --reserver_runtime_mem_mb 65536",
                    "decode": "--load_method fastsafetensors --max_seq_len 1100000 --enable_cuda_graph 0 --decode_capture_config '1,2,4,8' --act_type BF16 --tp_size 1 --dp_size 1 --ep_size 1 --world_size 1 --seq_size_per_block 256 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 16384 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 1 --cp_rotate_method PREFILL_CP --load_cache_timeout_ms 30000 --reserver_runtime_mem_mb 49152",
                },
                # The FP8 indexer's prefill score buffer is a dense
                # ``[chunk_rows, T] fp32`` block; at 1M context with CP=2 the
                # default 16384 rows want 16.44GiB on top of ~223GiB already
                # allocated, so halve it. expandable_segments recovers the ~29GiB
                # the caching allocator holds reserved-but-unallocated by then.
                envs={
                    "prefill": [
                        "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS=8192",
                        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    ],
                    "decode": [],
                },
                gpu_type=["L20D_TEST"],
            ),
            # Only DeepSeek-V4-Pro case. Deliberately plain — no CUDA graph, no PD
            # separation, no speculative decode, no page-RR. The advanced feature
            # matrix is covered on Flash above; this exists so a Pro-specific
            # regression (weight layout, expert count, rope/scaling config) cannot
            # ship unnoticed.
            #
            # Single-role CP=4 + EP=4 all-gather prefill topology, 4 GPUs. Runs on
            # L20D rather than GB200: Pro is ~865GB, so EP=4 needs ~216GB/rank,
            # over GB200's ~186GB but inside B300's ~288GB.
            smoke_test(
                name="v4_pro_cp4_ep4_basic_sm100",
                task_info="data/model/deepseek_v4/q_r_v4_pro_cp4_sm100_arm.json",
                smoke_args="--load_method scratch --max_seq_len 8192 --enable_cuda_graph 0 --act_type BF16 --tp_size 4 --dp_size 1 --ep_size 4 --moe_strategy mega_moe_se --world_size 4 --seq_size_per_block 256 --fp8_kv_cache 1 --use_deepep_moe 1 --use_deepep_low_latency 0 --cp_rotate_method ALL_GATHER --concurrency_limit 1 --max_context_batch_size 1 --reserver_runtime_mem_mb 20480",
                envs=["DG_JIT_CPP_STANDARD=20"],
                gpu_type=["L20D_TEST"],
            ),
            ":dispatcher_http_smoke",
            ":smoke_sm100_dsv4_flexlb_cache_affinity",
        ],
        tags = ["manual"],
    )

    _DSV4_FLEXLB_CACHE_AFFINITY_FIXTURE = "data/model/deepseek_v4/q_r_v4_flash_flexlb_cache_affinity_sm100_x86.json"

    native.filegroup(
        name = "flexlb_runtime_bundle",
        srcs = native.glob(["flexlb_runtime/**"], allow_empty = True),
    )

    native.py_test(
        name = "dispatcher_http_smoke",
        srcs = ["dispatcher_http_smoke_test.py"],
        main = "dispatcher_http_smoke_test.py",
        data = [":flexlb_runtime_bundle"],
        deps = SMOKE_FRAMEWORK_DEPS,
        timeout = "moderate",
        env = {"PYTHONNOUSERSITE": "1"},
        exec_properties = {"gpu": "L20D_TEST", "gpu_count": "1"},
        tags = ["manual", "smoke_case", "L20D_TEST"],
        legacy_create_init = 0,
    )

    _DSV4_FLEXLB_CACHE_AFFINITY_DATA = [
        _DSV4_FLEXLB_CACHE_AFFINITY_FIXTURE,
        ":flexlb_runtime_bundle",
        "//rtp_llm:sdk",
    ]

    # These cases boot a real FlexLB Spring Boot server. The CUDA13 x86 job builds
    # the uber-jar and a minimal Java 21 runtime into flexlb_runtime before Bazel
    # analysis; the filegroup above transfers both to the remote GPU sandbox.
    custom_smoke_test(
        name = "v4_flash_flexlb_shortest_ttft_cache_affinity_direct_2p2d_sm100_x86",
        main = "flexlb_cache_affinity_smoke_test.py",
        smoke_args = "--world_size 4",
        args = [
            "--fixture",
            _DSV4_FLEXLB_CACHE_AFFINITY_FIXTURE,
            "--strategy",
            "ShortestTtft",
            "--expected-decisions",
            "NO_CACHE_LEAD,NO_CACHE_LEAD,NO_CACHE_LEAD",
            "--max-extra-ttft-ms",
            "5000",
        ],
        data = _DSV4_FLEXLB_CACHE_AFFINITY_DATA,
        gpu_type = ["L20D_TEST"],
    )

    custom_smoke_test(
        name = "v4_flash_flexlb_cost_based_prefill_cache_affinity_direct_2p2d_sm100_x86",
        main = "flexlb_cache_affinity_smoke_test.py",
        smoke_args = "--world_size 4",
        args = [
            "--fixture",
            _DSV4_FLEXLB_CACHE_AFFINITY_FIXTURE,
            "--strategy",
            "CostBasedPrefill",
            "--expected-decisions",
            "NO_CACHE_LEAD,CACHE_LEADER,OVER_CAP",
            "--max-extra-ttft-ms",
            "300",
            "--prefill-cost-formula",
            "sum(computeTokens)+2*sum(hitCacheTokens)",
        ],
        data = _DSV4_FLEXLB_CACHE_AFFINITY_DATA,
        gpu_type = ["L20D_TEST"],
    )

    native.test_suite(
        name = "smoke_sm100_dsv4_flexlb_cache_affinity",
        tests = [
            ":v4_flash_flexlb_cost_based_prefill_cache_affinity_direct_2p2d_sm100_x86",
            ":v4_flash_flexlb_shortest_ttft_cache_affinity_direct_2p2d_sm100_x86",
        ],
    )
