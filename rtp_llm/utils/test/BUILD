load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "grpcio",
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
    name = "jit_cache_manager_test",
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

# Shared deps for the JIT cache lifecycle smoke tests (single source of truth).
_JIT_SMOKE_DEPS = [
    "//rtp_llm:dashscope",
    "//rtp_llm:fastapi",
    "//rtp_llm:jieba",
    "//rtp_llm:jit_cache_manager_lib",
    "//rtp_llm:json5",
    "//rtp_llm:openai",
    "//rtp_llm:partial_json_parser",
    "//rtp_llm:psutil",
    "//rtp_llm:pydantic",
    "//rtp_llm:safetensors",
    "//rtp_llm:sdk",
    "//rtp_llm:testlib",
    "//rtp_llm:tiktoken",
    "//rtp_llm:uvicorn",
    "//rtp_llm/test/utils:maga_server_manager",
]

# Shared-cache GPU smoke: a production watcher uploads a test-owned probe.
py_test(
    name = "jit_cache_deepseek_v2_lite",
    timeout = "eternal",
    srcs = ["jit_cache_smoke_test.py"],
    args = ["JitCacheSmokeTest.test_deepseek_v2_lite"],
    main = "jit_cache_smoke_test.py",
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
        "//rtp_llm:sdk",
        "//rtp_llm/test/smoke:deepseek_v2_lite_task_info",
    ],
    env = {
        "GPU_COUNT": "2",
        "PYTHONNOUSERSITE": "1",
    },
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "2",
    },
    tags = [
        "H20",
        "multi_device",
        "smoke_case",
    ],
    deps = _JIT_SMOKE_DEPS + ["//rtp_llm/cpp/model_rpc:grpcio"],
)

# ROCm lifecycle coverage intentionally omits the CUDA IPC weight-update path.
py_test(
    name = "jit_cache_qwen3_rocm",
    timeout = "eternal",
    srcs = ["jit_cache_smoke_test.py"],
    args = ["JitCacheSmokeTest.test_qwen3_rocm"],
    main = "jit_cache_smoke_test.py",
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
        "//rtp_llm:sdk",
        "//rtp_llm/test/smoke:qwen3_8b_task_info",
    ],
    env = {
        "GPU_COUNT": "1",
        "PYTHONNOUSERSITE": "1",
    },
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "1",
    },
    tags = [
        "MI308X-ROCM7",
        "rocm",
        "smoke_case",
    ],
    deps = _JIT_SMOKE_DEPS,
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
    name = "process_manager_test",
    srcs = [
        "process_manager_test.py",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
    ],
    exec_properties = {"gpu": "H20"},
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
    exec_properties = {"gpu": "H20"},
    deps = [
        ":grpcio",
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
    exec_properties = {"gpu": "A10"},
    deps = [
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:fastapi",
        "//rtp_llm:utils",
    ],
)
