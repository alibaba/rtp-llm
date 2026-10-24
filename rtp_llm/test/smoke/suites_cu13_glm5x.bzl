# GLM5x optimized profile: 1M limit (not a 1M workload assertion).
# CMP is decode-only: TP=1 is required; CP4 prefill uses GLM5_CP_OPT.
# Per rank: 1 GiB tiered MLA host pool + 4 GiB memory prefix cache;
# these are separate budgets, not enough to guarantee 1M residency.
# Set REMOTE_JIT_DIR to a trusted shared root in the test environment;
# an empty value disables remote restore/publish (local JIT still works).
load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")

def cu13_glm5x_suites():
    # One full-model case: CP4/EP4 prefill + DP4/EP4 decode (8 GPUs).
    # Keep MTP, FP8 KV, page-RR sharding, prefix reuse and decode graph enabled.
    smoke_test(
        env_inherit = ["REMOTE_JIT_DIR"],
        name = "glm5x_cu13_cp4_mtp_pd_smoke",
        task_info = "data/model/glm5/glm_5_fp8_cu13.json",
        smoke_args = {
                         "prefill": "--load_cache_timeout_ms 600000 --fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 32768 --max_seq_len 1048576 --role_type PREFILL --seq_size_per_block 512 --kernel_seq_size_per_block 64 --tp_size 4 --dp_size 1 --ep_size 4 --world_size 4 --warm_up 0 --reuse_cache 1 --enable_device_cache 1 --concurrency_limit 16 --quantization FP8_PER_BLOCK --moe_strategy mega_moe_se --use_deepep_moe 0 --use_deepep_low_latency 0 --use_all_gather 0 --sp_model_type glm_5_mtp --gen_num_per_cycle 5 --sp_type eagle --sp_checkpoint_path /mnt/hf3fs/3fs/models/GLM-5.3-FP4Moe-FP8ue8m0/ --sp_act_type bf16 --enable_cuda_graph 0 --cp_force_single_prefill 0 --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1 --load_method fastsafetensors --enable_memory_cache 1 --memory_cache_size_mb 4096 --manage_jit_cache 1",
                         "decode": "--load_cache_timeout_ms 600000 --fp8_kv_cache 1 --act_type BF16 --cache_store_rdma_mode 0 --use_local 1 --reserver_runtime_mem_mb 32768 --max_seq_len 1048576 --role_type DECODE --seq_size_per_block 512 --kernel_seq_size_per_block 64 --tp_size 1 --dp_size 4 --ep_size 4 --world_size 4 --warm_up 0 --reuse_cache 1 --enable_device_cache 1 --concurrency_limit 64 --quantization FP8_PER_BLOCK --moe_strategy mega_moe_se --use_all_gather 0 --sp_model_type glm_5_mtp --gen_num_per_cycle 5 --sp_type eagle --sp_checkpoint_path /mnt/hf3fs/3fs/models/GLM-5.3-FP4Moe-FP8ue8m0/ --sp_act_type bf16 --enable_cuda_graph 1 --decode_capture_config 1,4,8,16,24,32,64 --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 4 --load_method fastsafetensors --enable_memory_cache 1 --memory_cache_size_mb 4096 --manage_jit_cache 1"
                     },
        envs = {
                   "prefill": [
                       "CUDA_HOME=/usr/local/cuda",
                       "NVCC_PREPEND_FLAGS=-ccbin=/opt/rh/gcc-toolset-13/root/usr/bin/g++",
                       "DG_JIT_CPP_STANDARD=20",
                       "MOE_STRATEGY=mega_moe_se",
                       "USE_GATHER_PATH=1",
                       "RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING=0",
                       "THINK_END_TAG=</think>",
                       "RTP_LLM_GRPC_KEEPALIVE_TIME_MS=30000",
                       "RTP_LLM_GRPC_KEEPALIVE_TIMEOUT_MS=600000",
                       "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                       "PREFILL_CP_KV_CACHE_SHARDED=true",
                       "RTP_LLM_REQUIRE_CP_SHARDED_KV_CACHE=1",
                       "THINK_MODE=1",
                       "GLM5_CP_OPT=1",
                       "ENABLE_FUSE_KERNELS=1",
                       "ENABLE_FP32_LM_HEAD=0",
                       "RTP_LLM_DSA_MLA_HOST_CACHE_MB=1024",
                       "RTP_LLM_HOST_BLOCK_POOL_PREFAULT_THREADS=8",
                       "MANAGE_JIT_CACHE=1",
                       "RTP_LLM_GLM5_CMP=0"
                   ],
                   "decode": [
                       "CUDA_HOME=/usr/local/cuda",
                       "NVCC_PREPEND_FLAGS=-ccbin=/opt/rh/gcc-toolset-13/root/usr/bin/g++",
                       "DG_JIT_CPP_STANDARD=20",
                       "MOE_STRATEGY=mega_moe_se",
                       "RTP_LLM_GRPC_KEEPALIVE_TIME_MS=30000",
                       "RTP_LLM_GRPC_KEEPALIVE_TIMEOUT_MS=600000",
                       "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                       "RTP_LLM_STREAM_ASYNC=1",
                       "RTP_LLM_DROP_BROAD_SYNC=1",
                       "RTP_LLM_DEVICE_INPUT=1",
                       "ENABLE_FP32_LM_HEAD=0",
                       "PREFILL_CP_KV_CACHE_SHARDED=true",
                       "PREFILL_CP_SIZE=4",
                       "RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING=0",
                       "THINK_END_TAG=</think>",
                       "USE_GATHER_PATH=1",
                       "GLM5_CP_OPT=1",
                       "ENABLE_FUSE_KERNELS=1",
                       "RTP_LLM_DSA_MLA_HOST_CACHE_MB=1024",
                       "RTP_LLM_HOST_BLOCK_POOL_PREFAULT_THREADS=8",
                       "MANAGE_JIT_CACHE=1",
                       "RTP_LLM_GLM5_CMP=1"
                   ]
               },
        gpu_type = [
            "L20D_TEST"
        ],
    )
    native.test_suite(
        name = "smoke_cu13_glm5x",
        tests = [":glm5x_cu13_cp4_mtp_pd_smoke"],
        tags = ["manual"],
    )
