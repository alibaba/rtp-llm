load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def ppu_suites():
    # Explicit hardware regression for the migrated CP/reuse golden.
    smoke_test(
        name = "qwen35_moe_cp_reuse_tp2_ppu",
        task_info = "data/model/qwen35/qwen35_moe_bf16_tp2_ppu_cp_reuse.json",
        smoke_args = "--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --tp_size 2 --world_size 2 --cp_rotate_method ALL_GATHER --max_seq_len 32768 --kv_cache_mem_mb 4096 --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --load_method scratch",
        sleep_time_qr = 10,
        gpu_type = ["PPU-ZW810E"],
        tags = ["manual"],
    )

    # PPU (PPU-ZW810E) — All cases share basic framework features
    # ============================================================================

    # PPU basic framework features
    native.test_suite(
        name = "smoke_ppu_basic",
        tests = [
            smoke_test(
                name="streaming_ppu",
                task_info="data/model/qwen25/q_r_s_fp16_ppu.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="worker_status_reuse",
                task_info="data/model/qwen25/worker_status_reuse_cache_ppu.json",
                smoke_args="--reuse_cache 1 --seq_size_per_block 8 --act_type FP16 --warm_up 0",
                gpu_type=["PPU-ZW810E"],

            ),
            smoke_test(
                name="cache_status",
                task_info="data/model/qwen25/cache_status_ppu.json",
                # use scratch to avoid fastsafetensors pread failure on PPU NAS (errno 103 on 2GB blocks)
                smoke_args="--warm_up 0 --max_seq_len 16384 --ft_disable_custom_ar 1 --sp_type eagle --gen_num_per_cycle 4 --seq_size_per_block 256 --act_type FP16 --sp_model_type qwen_2-mtp --sp_checkpoint_path /mnt/nas1/mtp_reg/qwen2_14b_draft/ --tp_size 1 --load_method scratch",
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="calculate_loss_ppu",
                task_info="data/model/qwen25/q_r_s_loss_ppu.json",
                smoke_args="--act_type FP16 --warm_up 0",
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="cache_reuse_ppu",
                task_info="data/model/qwen25/q_r_reuse_device.json",
                sleep_time_qr=10,
                smoke_args="--reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --warm_up 0",
                gpu_type=["PPU-ZW810E"],

            ),
            # Tier selection is deployment-level; keep both original reuse goldens.
            smoke_test(
                name="memory_cache_reuse_ppu",
                task_info="data/model/qwen25/q_r_reuse_memory.json",
                sleep_time_qr=10,
                smoke_args="--reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --warm_up 0",
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="multi_task_ppu",
                task_info="data/model/qwen25/q_r_mt_ppu.json",
                smoke_args="--warm_up 0 --seq_size_per_block 8 --multi_task_prompt rtp_llm/test/smoke/data/model/qwen25/multitask.json --act_type FP16",
                gpu_type=["PPU-ZW810E"],

            ),
            smoke_test(
                name="multi_task_tp2_ppu",
                task_info="data/model/qwen25/q_r_mt_ppu.json",
                smoke_args="--warm_up 0 --seq_size_per_block 8 --multi_task_prompt rtp_llm/test/smoke/data/model/qwen25/multitask.json --act_type FP16 --tp_size 2",
                gpu_type=["PPU-ZW810E"],

            ),
            smoke_test(
                name="cache_edge_case",
                smoke_args="--reuse_cache 1 --act_type FP16 --enable_device_cache 1 --seq_size_per_block 4 --warm_up 0",
                gpu_type=["PPU-ZW810E"],
                task_info="data/model/qwen25/q_r_l20_cache_edge_case_1_ppu.json",

            ),
        ],
    )


    # PPU PD seperation
    native.test_suite(
        name = "smoke_ppu_pd",
        tests = [
            smoke_test(
                name="pd_tp2_to_tp1",
                task_info="data/model/qwen_7b/q_r_ppu_pd.json",
                smoke_args= {
                    "prefill": "--warm_up 0 --seq_size_per_block 64 --act_type fp16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --cache_store_thread_count 2 --reserver_runtime_mem_mb 8192 --tp_size 2",
                    "decode": "--warm_up 0 --seq_size_per_block 64 --act_type fp16 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --reserver_runtime_mem_mb 8192 --tp_size 1"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_tp1_to_tp2",
                task_info="data/model/qwen_7b/q_r_ppu_pd.json",
                smoke_args= {
                    "prefill": "--warm_up 0 --seq_size_per_block 64 --act_type fp16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --cache_store_thread_count 2 --reserver_runtime_mem_mb 8192 --tp_size 1",
                    "decode": "--warm_up 0 --seq_size_per_block 64 --act_type fp16 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --reserver_runtime_mem_mb 8192 --tp_size 2"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_tp2_to_tp2",
                task_info="data/model/qwen_7b/q_r_ppu_pd.json",
                smoke_args= {
                    "prefill": "--warm_up 0 --seq_size_per_block 64 --act_type fp16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --cache_store_thread_count 2 --reserver_runtime_mem_mb 8192 --tp_size 2",
                    "decode": "--warm_up 0 --seq_size_per_block 64 --act_type fp16 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --reserver_runtime_mem_mb 8192 --tp_size 2"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            # The three pd_reuse_* cases below run with --warm_up 1, unlike every
            # other pd_* case here. Measured on a ppu-4 ZW810E box: on a cold
            # start the prefill's *first* forward silently JIT-compiles its PPU
            # attention/rope kernels for ~19.5s (log goes quiet between
            # `[FA3PrefillPagedAttnOp] init` / `getRopeCache` and the next line).
            # That is inside the decode's --load_cache_timeout_ms window, so all
            # 24 per-layer cache-store loads expire against a prefill that has
            # published zero KV blocks -- the prefill logs 20+ of
            # `CacheStoreServiceImplContext.cpp runFailed(): ... blocks count: 0,
            # has watch func` and the request dies with
            # 8307_CACHE_STORE_LOAD_BUFFER_TIMEOUT. Raising the timeout does not
            # help: the loads are never answered, so they cost exactly the
            # deadline. --warm_up 1 moves the JIT off the request path.
            # Cold-start A/B on hardware: --warm_up 0 failed 8/8, --warm_up 1
            # passed 8/8 (pd_reuse_both) and 4/4 (pd_reuse_prefill).
            # Safe for these goldens because warm-up runs inside
            # LocalRpcServer::init, i.e. before NormalEngine::initCacheManager,
            # so it cannot populate the prefix cache the reuse_len goldens pin --
            # each deployment retains the original cold query and its DEVICE
            # or MEMORY reuse assertions, including the per-role PD counters.
            smoke_test(
                name="pd_reuse_both",
                task_info="data/model/qwen25/q_r_prefill_and_decode_reuse_device.json",
                sleep_time_qr=10,
                smoke_args= {
                    "prefill": "--warm_up 1 --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                    "decode": "--warm_up 1 --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type DECODE --reserver_runtime_mem_mb 4002"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_reuse_both_memory",
                task_info="data/model/qwen25/q_r_prefill_and_decode_reuse_memory.json",
                sleep_time_qr=10,
                smoke_args= {
                    "prefill": "--warm_up 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                    "decode": "--warm_up 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type DECODE --reserver_runtime_mem_mb 4002"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_reuse_prefill",
                task_info="data/model/qwen25/q_r_prefill_reuse_device.json",
                sleep_time_qr=10,
                smoke_args= {
                    "prefill": "--warm_up 1 --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                    "decode": "--warm_up 1 --reuse_cache 0 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type DECODE --reserver_runtime_mem_mb 4002"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_reuse_prefill_memory",
                task_info="data/model/qwen25/q_r_prefill_reuse_memory.json",
                sleep_time_qr=10,
                smoke_args= {
                    "prefill": "--warm_up 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                    "decode": "--warm_up 1 --reuse_cache 0 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type DECODE --reserver_runtime_mem_mb 4002"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_reuse_decode",
                task_info="data/model/qwen25/q_r_decode_reuse_device.json",
                sleep_time_qr=10,
                smoke_args= {
                    "prefill": "--warm_up 1 --reuse_cache 0 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                    "decode": "--warm_up 1 --reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type DECODE --reserver_runtime_mem_mb 4002"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_reuse_decode_memory",
                task_info="data/model/qwen25/q_r_decode_reuse_memory.json",
                sleep_time_qr=10,
                smoke_args= {
                    "prefill": "--warm_up 1 --reuse_cache 0 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                    "decode": "--warm_up 1 --reuse_cache 1 --enable_device_cache 0 --enable_memory_cache 1 --memory_cache_size_mb 1024 --seq_size_per_block 8 --act_type FP16 --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 20000 --role_type DECODE --reserver_runtime_mem_mb 4002"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_multi_task",
                # PD deployments report aux_info.pd_sep=true; dedicated golden.
                task_info="data/model/qwen25/q_r_mt_pd.json",
                smoke_args= {
                    "prefill": "--warm_up 0 --act_type FP16 --seq_size_per_block 8 --multi_task_prompt rtp_llm/test/smoke/data/model/qwen25/multitask.json --cache_store_rdma_mode 0 --use_local 1 --role_type PREFILL --reserver_runtime_mem_mb 4002",
                    "decode": "--warm_up 0 --act_type FP16 --seq_size_per_block 8 --multi_task_prompt rtp_llm/test/smoke/data/model/qwen25/multitask.json --cache_store_rdma_mode 0 --use_local 1 --role_type DECODE --reserver_runtime_mem_mb 4002"
                },
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="pd_multi_task_tp2",
                task_info="data/model/qwen25/q_r_mt_pd.json",
                smoke_args= {
                    "prefill": "--warm_up 0 --act_type FP16 --seq_size_per_block 8 --multi_task_prompt rtp_llm/test/smoke/data/model/qwen25/multitask.json --cache_store_rdma_mode 0 --use_local 1 --role_type PREFILL --tp_size 2 --reserver_runtime_mem_mb 4002",
                    "decode": "--warm_up 0 --act_type FP16 --seq_size_per_block 8 --multi_task_prompt rtp_llm/test/smoke/data/model/qwen25/multitask.json --cache_store_rdma_mode 0 --use_local 1 --role_type DECODE --tp_size 2 --reserver_runtime_mem_mb 4002"
                },
                gpu_type=["PPU-ZW810E"],
            ),
        ],
    )


    # Qwen3.5-MoE basic single-node coverage: DeepEP / TP=8 MoE modes + P/D TP=4.
    # (Colocated MoE-mode + P/D tests, moved out of smoke_ppu_pd.)
    native.test_suite(
        name = "smoke_ppu_qwen35_basic",
        tests = [
            # Qwen3.5-MoE DP=4 TP=2 DeepEP LL + CudaGraph (8 卡, EP=8)
            smoke_test(
                name="qwen35_moe_dp4_tp2_deepep_ll_cg_ppu",
                task_info="data/model/qwen35/qwen35_moe_bf16_tp4_ppu.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --reserver_runtime_mem_mb 4096 --tp_size 2 --dp_size 4 --world_size 8 --use_deepep_moe 1 --use_deepep_low_latency 1 --enable_cuda_graph 1 --decode_capture_config 1,2,3,4",
                envs=["DEEPEP_DISABLE_NETWORK=1"],
                gpu_type=["PPU-ZW810E"],
            ),
            # TP=8 DeepEP Normal + DeepGEMM BF16 (8 卡, EP=8, 显式 opt-in)
            smoke_test(
                name="qwen35_moe_tp8_deepep_normal_deepgemm_ppu",
                task_info="data/model/qwen35/qwen35_moe_bf16_tp4_ppu.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --reserver_runtime_mem_mb 4096 --tp_size 8 --world_size 8 --use_deepep_moe 1 --use_deepep_low_latency 0 --moe_strategy no_quant_dp_normal_deepgemm",
                gpu_type=["PPU-ZW810E"],
            ),
            # Qwen3.5-MoE PD 分离: 混合架构 (30 LinearAttn + 10 MHA + MoE), TP=4 both sides
            # Decode 侧开全部优化 (CudaGraph + F1 moe_align + T1 kspb=128)
            smoke_test(
                name="qwen35_moe_pd_tp4_to_tp4_ppu",
                task_info="data/model/qwen35/qwen35_moe_bf16_tp4_ppu_pd.json",
                smoke_args= {
                    "prefill": "--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --reserver_runtime_mem_mb 4096 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --tp_size 4",
                    "decode": "--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --reserver_runtime_mem_mb 4096 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --tp_size 4 --enable_cuda_graph 1 --decode_capture_config 1,2,3,4"
                },
                gpu_type=["PPU-ZW810E"],
            ),
        ],
    )

    # Qwen3.5-MoE single-node functional coverage: MTP + reuse-cache
    # Mirrors GPU suites_h20_oss.bzl smoke_h20_next (next_mtp_basic / next_long_reuse_memcache)
    native.test_suite(
        name = "smoke_ppu_qwen35_func",
        tests = [
            # MTP (Multi-Token Prediction): Qwen3.5-35B-A3B ships a built-in 1-layer MTP,
            # so sp_checkpoint_path points to the main weights (mtp_use_dedicated_embeddings=false).
            smoke_test(
                name="qwen35_moe_mtp_tp4_ppu",
                task_info="data/model/qwen35/qwen35_moe_bf16_tp4_ppu_mtp.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --tp_size 4 --world_size 4 --reserver_runtime_mem_mb 4096 --max_seq_len 12800 --sp_type eagle --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B --sp_act_type bf16",
                gpu_type=["PPU-ZW810E"],
            ),
            # Production-shaped P/D coverage for MTP + CUDA Graph. Keeping the
            # two TP=4 workers separate avoids the colocated draft-capture block
            # table trap while exercising target verify, draft decode, and draft
            # prefill graphs on the decode worker.
            smoke_test(
                name="qwen35_moe_mtp_cg_tp4_ppu",
                task_info="data/model/qwen35/qwen35_moe_bf16_tp4_ppu_mtp.json",
                smoke_args={
                    "prefill": "--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --tp_size 4 --world_size 4 --reserver_runtime_mem_mb 4096 --max_seq_len 12800 --concurrency_limit 16 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --sp_type eagle --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B --sp_act_type bf16",
                    "decode": "--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --tp_size 4 --world_size 4 --reserver_runtime_mem_mb 4096 --max_seq_len 12800 --concurrency_limit 16 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --sp_type eagle --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-35B-A3B --sp_act_type bf16 --enable_cuda_graph 1 --decode_capture_config 1,2,3,4",
                },
                envs={
                    "prefill": ["LOAD_METHOD=scratch"],
                    "decode": ["LOAD_METHOD=scratch"],
                },
                gpu_type=["PPU-ZW810E"],
            ),
            # reuse-cache / prefix caching: second identical query must hit memory cache.
            # Use --kv_cache_mem_mb to bypass automatic memory check (Rank 0 OOM with memory_cache allocation)
            smoke_test(
                name="qwen35_moe_reuse_tp4_ppu",
                task_info="data/model/qwen35/qwen35_moe_bf16_tp4_ppu_reuse.json",
                sleep_time_qr=10,
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --tp_size 4 --world_size 4 --max_seq_len 2048 --kv_cache_mem_mb 1024 --reuse_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 512",
                gpu_type=["PPU-ZW810E"],
            ),
        ],
    )


    # Qwen3.5-397B-A17B W8A8 INT8: mirrors the shape actually deployed online
    # (w8a8_int8 DeepEP strategies, TP8, MTP on the low-latency DeepGEMM path).
    #
    # NOT wired into maga_model_smoke / maga_model_smoke_light: the 397B INT8
    # weights are not available on CI runners. Every case is tagged "manual" so a
    # wildcard expansion (//...) cannot pick them up either. Run locally with an
    # explicit target on a host that has /mnt/nas1/hf/Qwen3.5-397B-A17B-INT8.
    #
    # Golden status on this branch:
    #   qwen35_moe_w8a8_int8_mtp_tp8_dp2_ppu     verified PASSED locally
    #   qwen35_moe_w8a8_int8_mtp_cg_tp8_dp2_ppu  golden inherited from
    #   qwen35_moe_w8a8_int8_pd_tp8_to_tp8_ppu   feature/qwen35-ppu-internal-clean;
    #     both diverge at token level here (server + P/D path work, text differs).
    #     Regenerate with --config=rewrite_smoke on a host with 16 free PPUs.
    native.test_suite(
        name = "smoke_ppu_qwen35_w8a8_func",
        tags = ["manual"],
        tests = [
            smoke_test(
                name="qwen35_moe_w8a8_int8_mtp_tp8_dp1_ppu",
                task_info="data/model/qwen35/qwen35_moe_w8a8_int8_tp8_dp2_ppu_mtp.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --tp_size 8 --dp_size 1 --world_size 8 --reserver_runtime_mem_mb 4096 --max_seq_len 12800 --use_deepep_moe 1 --use_deepep_low_latency 1 --moe_strategy w8a8_int8_ep_low_latency_deepgemm --sp_type eagle --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-397B-A17B-INT8 --sp_act_type bf16",
                envs=["LOAD_METHOD=scratch"],
                tags=["manual"],
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="qwen35_moe_w8a8_int8_mtp_tp8_dp2_ppu",
                task_info="data/model/qwen35/qwen35_moe_w8a8_int8_tp8_dp2_ppu_mtp.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --tp_size 8 --dp_size 2 --world_size 16 --reserver_runtime_mem_mb 4096 --max_seq_len 12800 --use_deepep_moe 1 --use_deepep_low_latency 1 --moe_strategy w8a8_int8_ep_low_latency_deepgemm --sp_type eagle --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-397B-A17B-INT8 --sp_act_type bf16",
                envs=["LOAD_METHOD=scratch"],
                tags=["manual"],
                gpu_type=["PPU-ZW810E"],
            ),
            smoke_test(
                name="qwen35_moe_w8a8_int8_mtp_cg_tp8_dp2_ppu",
                task_info="data/model/qwen35/qwen35_moe_w8a8_int8_tp8_dp2_ppu_mtp_cg.json",
                smoke_args="--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --tp_size 8 --dp_size 2 --world_size 16 --reserver_runtime_mem_mb 4096 --max_seq_len 12800 --use_deepep_moe 1 --use_deepep_low_latency 1 --moe_strategy w8a8_int8_ep_low_latency_deepgemm --sp_type eagle --sp_model_type qwen35_moe_mtp --gen_num_per_cycle 4 --sp_checkpoint_path /mnt/nas1/hf/Qwen3.5-397B-A17B-INT8 --sp_act_type bf16 --enable_cuda_graph 1 --decode_capture_config 1,2,3,4",
                envs=["LOAD_METHOD=scratch"],
                tags=["manual"],
                gpu_type=["PPU-ZW810E"],
            ),
            # P/D disaggregation, INT8, no RDMA (cache_store_rdma_mode 0):
            # prefill on the Normal DeepEP path, decode on low-latency + cuda graph.
            smoke_test(
                name="qwen35_moe_w8a8_int8_pd_tp8_to_tp8_ppu",
                task_info="data/model/qwen35/qwen35_moe_w8a8_int8_tp8_ppu_pd.json",
                smoke_args={
                    "prefill": "--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --reserver_runtime_mem_mb 4096 --role_type PREFILL --cache_store_rdma_mode 0 --use_local 1 --tp_size 8 --dp_size 1 --ep_size 8 --world_size 8 --use_deepep_moe 1 --use_deepep_low_latency 0 --moe_strategy w8a8_int8_dp_normal_deepgemm",
                    "decode": "--warm_up 0 --act_type BF16 --seq_size_per_block 2048 --reserver_runtime_mem_mb 4096 --role_type DECODE --cache_store_rdma_mode 0 --use_local 1 --tp_size 8 --dp_size 1 --ep_size 8 --world_size 8 --use_deepep_moe 1 --use_deepep_low_latency 1 --moe_strategy w8a8_int8_ep_low_latency_deepgemm --enable_cuda_graph 1 --decode_capture_config 1,2,3,4",
                },
                envs={
                    "prefill": ["LOAD_METHOD=scratch"],
                    "decode": ["LOAD_METHOD=scratch"],
                },
                tags=["manual"],
                gpu_type=["PPU-ZW810E"],
            ),
        ],
    )
