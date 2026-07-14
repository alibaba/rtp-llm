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
