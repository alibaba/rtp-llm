load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "grpcio",
    "packaging",
])

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
    ]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "jit_cache_manager_test_lib",
    srcs = ["jit_cache_manager_test.py"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:jit_cache_manager_lib",
        "//rtp_llm/metrics",
    ],
)

py_test(
    name = "jit_cache_manager_test",
    srcs = [
        "jit_cache_manager_test.py",
    ],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:jit_cache_manager_lib",
        "//rtp_llm/metrics",
    ],
)

py_test(
    name = "jit_cache_remote_integration_test",
    timeout = "long",
    srcs = [
        "jit_cache_remote_integration_test.py",
    ],
    env = {
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "GPU_COUNT": "1",
    },
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = [
        "H20",
        "open_skip",
    ],
    deps = [
        ":jit_cache_manager_test_lib",
        ":packaging",
        "//rtp_llm:apache-tvm-ffi",
        "//rtp_llm:deep_gemm",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:jit_cache_manager_lib",
        "//rtp_llm:torch",
        "//rtp_llm:triton",
        "//rtp_llm/models_py/triton_kernels/autotune_cache",
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
