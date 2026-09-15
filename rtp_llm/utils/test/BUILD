
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
    size = "large",
    srcs = ["jit_cache_manager_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:sdk",
        "//rtp_llm:torch",
        "//rtp_llm:utils",
        "//rtp_llm/model_loader/tipc",
    ],
)

py_test(
    name = "ckpt_database_test",
    srcs = [
        "ckpt_database_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:config",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:lora"
    ],
    data = [
        "//rtp_llm/utils/test:testdata",
    ]
)

py_test(
    name = "ckpt_database_fastsafetensors_multirank_test",
    size = "medium",
    timeout = "moderate",
    srcs = ["ckpt_database_fastsafetensors_multirank_test.py"],
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
    exec_properties = {"gpu": "H20", "gpu_count": "2"},
    tags = ["multi_device"],
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
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
    name = "dsa_indexing_test",
    srcs = ["dsa_indexing_test.py"],
    deps = ["//rtp_llm:utils"],
)


py_test(
    name = "util_test",
    srcs = [
        "util_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:pynvml",
        "//rtp_llm:psutil",
        "//rtp_llm:fastapi",
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
    deps = [
        "//rtp_llm:utils",
    ],
    data = [
        "//:th_transformer",
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    exec_properties = {'gpu':'H20'}
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
    deps = [
        "//rtp_llm:utils",
        ":grpcio",
    ],
    data = [
        "//:th_transformer",
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    exec_properties = {'gpu':'H20'}
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
    name = "scr_template_utils_test",
    srcs = [
        "scr_template_utils_test.py",
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
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:fastapi",
        "//rtp_llm:_ft_pickler",
    ],
    data = [
        "//:th_transformer",
        "//:th_transformer_config",
        "//:rtp_compute_ops",
    ],
    exec_properties = {'gpu':'A10'}
)

py_test(
    name = "scr_local_comm_test",
    srcs = ["scr_local_comm_test.py"],
    deps = ["//rtp_llm:utils"],
)

py_test(
    name = "scr_transport_boundary_test",
    srcs = ["scr_transport_boundary_test.py"],
    data = ["//:th_transformer_config"],
    deps = ["//rtp_llm:config", "//rtp_llm:config_ops", "//rtp_llm/distribute:distribute"],
)

py_test(
    name = "scr_endpoint_provider_test",
    srcs = ["scr_endpoint_provider_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
        "//rtp_llm/distribute:distribute",
    ],
)

py_test(
    name = "scr_endpoint_config_test",
    srcs = ["scr_endpoint_config_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:sdk",
        "//rtp_llm:structure",
        "//rtp_llm:utils",
        "//rtp_llm/access_logger:access_logger",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/distribute:distribute",
        "//rtp_llm/server:server",
    ],
)

py_test(
    name = "scr_template_lifecycle_test",
    srcs = ["scr_template_lifecycle_test.py"],
    deps = ["//rtp_llm:utils"],
)

py_test(
    name = "scr_restore_context_test",
    srcs = ["scr_restore_context_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
        "//rtp_llm/distribute:distribute",
    ],
)

py_test(
    name = "scr_service_startup_test",
    srcs = ["scr_service_startup_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:sdk",
        "//rtp_llm:structure",
        "//rtp_llm:utils",
        "//rtp_llm/access_logger:access_logger",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/distribute:distribute",
        "//rtp_llm/server:server",
    ],
)

py_test(
    name = "scr_kmonitor_lifecycle_test",
    srcs = ["scr_kmonitor_lifecycle_test.py"],
    deps = ["//rtp_llm:utils", "//rtp_llm/aios/kmonitor:kmonitor_py"],
)

py_test(
    name = "scr_runtime_fixup_test",
    srcs = ["scr_runtime_fixup_test.py"],
    deps = ["//rtp_llm:utils", "//rtp_llm/aios/kmonitor:kmonitor_py"],
)

py_test(
    name = "scr_restore_env_file_test",
    srcs = ["scr_restore_env_file_test.py"],
    deps = ["//rtp_llm:utils"],
)

py_test(
    name = "scr_advertise_ip_test",
    srcs = ["scr_advertise_ip_test.py"],
    deps = ["//rtp_llm:utils"],
)

py_test(
    name = "scr_pd_advertisement_test",
    srcs = ["scr_pd_advertisement_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
        "//rtp_llm/distribute:distribute",
    ],
)

py_test(
    name = "scr_native_logger_test",
    srcs = ["scr_native_logger_test.py"],
    data = ["//rtp_llm/cpp/utils:scr_logger_test_sources"],
)
