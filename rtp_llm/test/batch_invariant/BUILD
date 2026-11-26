load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

py_test(
    name = "config_test",
    srcs = ["config_test.py"],
    env = {"GPU_COUNT": "0"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "m11_gdn_test",
    srcs = ["m11_gdn_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    deps = ["//rtp_llm:testlib"],
)

py_test(
    name = "m11_ops_test",
    srcs = ["m11_ops_test.py", "m11_gdn_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
    deps = ["//rtp_llm:testlib"],
)

py_library(
    name = "service_test_lib",
    srcs = ["service_test.py"],
    deps = ["//rtp_llm:testlib"],
)

custom_smoke_test(
    name = "qwen4_m11_mxfp4_sm10x",
    main = "m11_service_test.py",
    smoke_args = " ".join([
        "--load_method scratch --act_type BF16 --role_type PDFUSION",
        "--tp_size 1 --dp_size 8 --ep_size 8 --world_size 8",
        "--moe_strategy mega_moe --use_deepep_moe 0 --vit_separation 2",
        "--max_seq_len 8192 --max_batch_tokens_size 8192",
        "--seq_size_per_block 128 --kernel_seq_size_per_block 128 --ssm_state_dtype fp32",
        "--kv_cache_mem_mb 8192 --linear_pool_blocks 8 --reuse_cache 0",
        "--enable_fp32_lm_head 0 --enable_cuda_graph 0 --warm_up 0",
        "--enable_batch_record 1 --record_decode_sample_rate 1",
    ]),
    gpu_type = ["SM103_CU13"],
    data = ["//rtp_llm:sdk"],
    deps = [":service_test_lib", "//rtp_llm/test/smoke:qwen4_m11_smoke"],
    test_envs = {"RTP_TEST_WATCHDOG_SECONDS": "7200"},
)
