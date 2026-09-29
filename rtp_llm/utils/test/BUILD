load("@arch_config//:arch_select.bzl", "requirement")
load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

requirement([
    "grpcio",
    "protobuf",
])

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
    ]),
    visibility = ["//visibility:public"],
)

py_test(
    name = "swizzle_utils_test",
    srcs = [
        "swizzle_utils_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "jit_cache_manager_test",
    size = "large",
    srcs = [
        "jit_cache_manager_test.py",
    ],
    data = [
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:jit_cache_manager_lib",
        "//rtp_llm:sdk",
        "//rtp_llm:torch",
        "//rtp_llm/model_loader/tipc",
    ],
)

# Extra deps beyond the smoke framework for the JIT cache lifecycle smokes.
_JIT_SMOKE_EXTRA_DEPS = [
    "//rtp_llm:jit_cache_manager_lib",
    "//rtp_llm:safetensors",
    "//rtp_llm:sdk",
]

# Aone injects REMOTE_JIT_DIR; direct runs must provide it explicitly to verify
# the shared-cache lifecycle instead of passing on a private cold start.
custom_smoke_test(
    name = "jit_cache_deepseek_v2_lite",
    args = ["JitCacheSmokeTest.test_deepseek_v2_lite"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
        "//rtp_llm:sdk",
        "//rtp_llm/test/smoke:deepseek_v2_lite_task_info",
    ],
    gpu_type = ["H20_CU13"],
    main = "jit_cache_smoke_test.py",
    smoke_args = "--warm_up 0 --hack_layer_num 2 --load_method scratch --test_block_num 100 --act_type BF16 --quantization FP8_PER_BLOCK --seq_size_per_block 64 --tp_size 2 --reuse_cache 1",
    tags = ["multi_device"],
    deps = _JIT_SMOKE_EXTRA_DEPS + [":grpcio"],
)

# ROCm lifecycle coverage intentionally omits the CUDA IPC weight-update path.
custom_smoke_test(
    name = "jit_cache_qwen3_rocm",
    args = ["JitCacheSmokeTest.test_qwen3_rocm"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
        "//rtp_llm:sdk",
        "//rtp_llm/test/smoke:qwen3_8b_task_info",
    ],
    gpu_type = ["MI308X-ROCM7"],
    main = "jit_cache_smoke_test.py",
    smoke_args = "--warm_up 0 --use_swizzleA 1 --use_asm_pa 1 --disable_flashinfer_native 1 --use_aiter_pa 1 --seq_size_per_block 16 --act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000",
    tags = ["rocm"],
    deps = _JIT_SMOKE_EXTRA_DEPS,
)

py_test(
    name = "ckpt_database_test",
    srcs = [
        "ckpt_database_test.py",
    ],
    data = [
        "//rtp_llm/utils/test:testdata",
    ],
    deps = [
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:config",
        "//rtp_llm:lora",
        "//rtp_llm:utils",
    ],
)

py_test(
    # Verifies load_tensor_slice / copy-out on a real ROCm build (hip asserted).
    name = "ckpt_database_test_rocm",
    srcs = ["ckpt_database_test.py"],
    args = ["HandleRecyclingTest"],
    data = ["//rtp_llm/utils/test:testdata"],
    env = {"RTP_LLM_REQUIRE_ACCELERATOR": "1"},
    exec_properties = {"gpu": "MI308X-ROCM7"},
    main = "ckpt_database_test.py",
    tags = ["rocm"],
    deps = [
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:config",
        "//rtp_llm:lora",
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "fuser_test",
    srcs = [
        "fuser_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "util_test",
    srcs = [
        "util_test.py",
    ],
    deps = [
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:fastapi",
        "//rtp_llm:psutil",
        "//rtp_llm:pynvml",
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "warmup_test",
    srcs = ["warmup_test.py"],
    deps = ["//rtp_llm:warmup"],
)

py_test(
    name = "hot_hook_runtime_test",
    srcs = [
        "hot_hook_runtime_test.py",
        "hot_hook_target.py",
    ],
    deps = [
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "process_manager_test",
    srcs = [
        "process_manager_test.py",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "grpc_host_channel_pool_test",
    srcs = [
        "grpc_host_channel_pool_test.py",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        ":grpcio",
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "grpc_client_wrapper_test",
    srcs = ["grpc_client_wrapper_test.py"],
    deps = [
        ":grpcio",
        ":protobuf",
        "//rtp_llm:utils",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
        "//rtp_llm/metrics",
    ],
)

py_test(
    name = "oom_diag_test",
    srcs = [
        "oom_diag_test.py",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "duplicated_kv_test",
    srcs = [
        "duplicated_kv_test.py",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
    ],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = [
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:fastapi",
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "backend_registry_test",
    srcs = [
        "backend_registry_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
    ],
)
