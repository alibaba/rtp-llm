load("//bazel:arch_select.bzl", "requirement")

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
        "jit_cache_smoke_test.py",
    ],
    data = [
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:jit_cache_manager_lib",
    ],
)

py_test(
    name = "jit_cache_deepseek_v2_lite",
    timeout = "eternal",
    srcs = [
        "jit_cache_smoke_test.py",
    ],
    args = [
        "JitCacheSmokeTest.test_deepseek_v2_lite",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
        "//rtp_llm:sdk",
    ],
    env = {
        "PYTHONNOUSERSITE": "1",
    },
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    main = "jit_cache_smoke_test.py",
    tags = [
        "H20",
        "manual",
        "open_skip",
    ],
    deps = [
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
        "//rtp_llm/cpp/model_rpc:grpcio",
        "//rtp_llm/test/utils:maga_server_manager",
    ],
)

test_suite(
    name = "smoke_jit_cache_all_components",
    tests = [
        ":jit_cache_deepseek_v2_lite",
    ],
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
    name = "multimodal_util_test",
    srcs = [
        "multimodal_util_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:pynvml",
        "//rtp_llm:Pillow",
        "//rtp_llm:oss2",
        "//rtp_llm:pyOpenSSL",
        "//rtp_llm:config_ops"
    ],
    data = [
        "//:th_transformer",
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    exec_properties = {'gpu':'H20'}
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

load("//bazel:arch_select.bzl", "requirement")

requirement([
    "grpcio",
])

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
    name = "oom_diag_test",
    srcs = [
        "oom_diag_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
    ],
    data = [
        "//:th_transformer",
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    exec_properties = {'gpu':'A10'}
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
