load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")
load("@arch_config//:arch_select.bzl", "cuda13_x86_test_gpu")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "smoke_lib",
    srcs = [
        "grammar_smoke.py",
        "logprobs_smoke.py",
        "request_smoke.py",
        "text_smoke.py",
    ],
    imports = ["."],
)

custom_smoke_test(
    name = "kimi_k31_single",
    main = "kimi_k31_single_test.py",
    gpu_type = [cuda13_x86_test_gpu()],
    smoke_args = " ".join([
        "--role_type PDFUSION --tp_size 8 --dp_size 1 --ep_size 8 --world_size 8 --local_world_size 8",
        "--frontend_server_count 1 --load_method fastsafetensors --act_type BF16",
        "--quantization KIMI_K31_NVFP4 --moe_strategy mega_moe_se",
        "--sp_type mtp --sp_model_type kimi_k31_mtp --sp_act_type BF16 --gen_num_per_cycle 3",
        "--max_seq_len 2097152 --max_batch_tokens_size 262144 --max_batch_tokens_without_cache 65536 --prefill_chunk_size 65536",
        "--max_context_batch_size 16 --concurrency_limit 64 --warm_up 0 --model_warm_up 0",
        "--enable_cuda_graph 1 --decode_capture_config 1,2,4,8,16,32,64 --enable_linear_replay 1",
        # K3.1 uses local PageRR cache sharding in PDFUSION.
        "--prefill_cp_kv_cache_sharded 1",
        "--fp8_kv_cache 1 --seq_size_per_block 4096 --kernel_seq_size_per_block 128 --ssm_state_dtype fp32",
        "--reuse_cache 1 --enable_device_cache 1 --enable_memory_cache 1 --memory_cache_size_mb 32768 --enable_remote_cache 0 --cache_store_rdma_mode 0",
        # Reserve room for long-context indexer scoring and the draft KV projection.
        "--prefill_kv_workspace_budget_bytes 6442450944 --reserver_runtime_mem_mb 26624 --backend_post_frontend_drain_seconds 10",
    ]),
    test_envs = {
        # SM103 can fail on the first BS64 Graph replay with NCCL graph registration.
        "NCCL_GRAPH_REGISTER": "0",
        "RTP_TEST_WATCHDOG_SECONDS": "7200",
    },
    deps = [
        ":smoke_lib",
        "//rtp_llm:sdk",
    ] + select({
        "@//:using_cuda13_x86": ["//rtp_llm:fastsafetensors", "//rtp_llm:fast-safetensors"],
        "//conditions:default": [],
    }),
)

_OFFLINE_TESTS = [
    "test_grammar_smoke",
    "test_logprobs_smoke",
    "test_orthogonal_smoke",
    "test_request_smoke",
    "test_text_smoke_capped",
]

[
    py_test(
        name = name,
        srcs = [name + ".py"],
        deps = [":smoke_lib"],
        env = {"GPU_COUNT": "0"},
    )
    for name in _OFFLINE_TESTS
]

test_suite(
    name = "offline_tests",
    tests = [":" + name for name in _OFFLINE_TESTS],
)
