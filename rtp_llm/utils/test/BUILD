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
    name = "database_fastsafetensors_region_test",
    srcs = [
        "database_fastsafetensors_region_test.py",
    ],
    deps = [
        "//rtp_llm:lora",
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
    name = "grpc_client_wrapper_test",
    srcs = ["grpc_client_wrapper_test.py"],
    deps = [
        ":grpcio",
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
    name = "nccl_memory_test",
    srcs = [
        "nccl_memory_test.py",
    ],
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
    exec_properties = {"gpu": "A10"},
    deps = [
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:fastapi",
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "ckpt_database_fastsafetensors_multirank_arm_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["ckpt_database_fastsafetensors_multirank_test.py"],
    main = "ckpt_database_fastsafetensors_multirank_test.py",
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:config",
        "//rtp_llm:fast-safetensors",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:lora",
        "//rtp_llm:ops",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:utils",
    ],
    env = {
        "GPU_COUNT": "2",
        "WORLD_SIZE": "2",
    },
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "2"},
    tags = ["multi_device", "SM100_ARM"],
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
