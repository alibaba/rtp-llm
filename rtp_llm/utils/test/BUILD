load("@arch_config//:arch_select.bzl", "requirement")
load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

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
    main = "jit_cache_smoke_test.py",
    smoke_args = "--warm_up 0 --hack_layer_num 2 --load_method scratch --test_block_num 100 --act_type BF16 --quantization FP8_PER_BLOCK --seq_size_per_block 64 --tp_size 2 --reuse_cache 1",
    args = ["JitCacheSmokeTest.test_deepseek_v2_lite"],
    gpu_type = ["H20"],
    tags = ["multi_device"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
        "//rtp_llm:sdk",
        "//rtp_llm/test/smoke:deepseek_v2_lite_task_info",
    ],
    deps = _JIT_SMOKE_EXTRA_DEPS + [":grpcio"],
)

# ROCm lifecycle coverage intentionally omits the CUDA IPC weight-update path.
custom_smoke_test(
    name = "jit_cache_qwen3_rocm",
    main = "jit_cache_smoke_test.py",
    smoke_args = "--warm_up 0 --use_swizzleA 1 --use_asm_pa 1 --disable_flashinfer_native 1 --use_aiter_pa 1 --seq_size_per_block 16 --act_type BF16 --test_block_num 1000 --reserver_runtime_mem_mb 70000",
    args = ["JitCacheSmokeTest.test_qwen3_rocm"],
    gpu_type = ["MI308X-ROCM7"],
    tags = ["rocm"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//:th_transformer_config",
        "//rtp_llm:sdk",
        "//rtp_llm/test/smoke:qwen3_8b_task_info",
    ],
    deps = _JIT_SMOKE_EXTRA_DEPS,
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
    name = "ckpt_database_fastsafetensors_contract_test",
    srcs = ["ckpt_database_test.py"],
    main = "ckpt_database_test.py",
    args = ["InstalledFastsafetensorsContractTest"],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:config",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:lora",
        "//rtp_llm:fastsafetensors",
    ],
    data = ["//rtp_llm/utils/test:testdata"],
    env = {"RTP_LLM_EXPECT_FASTSAFETENSORS_TIER": "per-expert"},
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "@//:using_cuda13_x86": [],
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
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
    exec_properties = {"gpu": "H20", "gpu_count": "2"},
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
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
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "2"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "rtp_llm_cuda12_9_wheel_metadata_test",
    srcs = ["rtp_llm_wheel_metadata_test.py"],
    main = "rtp_llm_wheel_metadata_test.py",
    args = [
        "cuda12_9_x86",
        "$(location //deps:requirements_lock_torch_gpu_cuda12_9.txt)",
        "$(location //rtp_llm:rtp_llm_whl)",
    ],
    data = [
        "//deps:requirements_lock_torch_gpu_cuda12_9.txt",
        "//rtp_llm:rtp_llm_whl",
    ],
    target_compatible_with = select({
        "@//:using_cuda12_9_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "rtp_llm_cuda13_x86_wheel_metadata_test",
    srcs = ["rtp_llm_wheel_metadata_test.py"],
    main = "rtp_llm_wheel_metadata_test.py",
    args = [
        "cuda13_x86",
        "$(location //deps:requirements_lock_torch_gpu_cuda13.txt)",
        "$(location //rtp_llm:rtp_llm_whl)",
    ],
    data = [
        "//deps:requirements_lock_torch_gpu_cuda13.txt",
        "//rtp_llm:rtp_llm_whl",
    ],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "rtp_llm_cuda13_arm_wheel_metadata_test",
    srcs = ["rtp_llm_wheel_metadata_test.py"],
    main = "rtp_llm_wheel_metadata_test.py",
    args = [
        "cuda13_arm",
        "$(location //deps:requirements_lock_cuda13_arm.txt)",
        "$(location //rtp_llm:rtp_llm_whl)",
    ],
    data = [
        "//deps:requirements_lock_cuda13_arm.txt",
        "//rtp_llm:rtp_llm_whl",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    # Verifies load_tensor_slice / copy-out on a real ROCm build (hip asserted).
    name = "ckpt_database_test_rocm",
    srcs = ["ckpt_database_test.py"],
    main = "ckpt_database_test.py",
    args = ["HandleRecyclingTest"],
    deps = ["//rtp_llm:utils", "//rtp_llm:config", "//rtp_llm:_ft_pickler", "//rtp_llm:lora"],
    data = ["//rtp_llm/utils/test:testdata"],
    env = {"RTP_LLM_REQUIRE_ACCELERATOR": "1"},
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X-ROCM7'},
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
        "//rtp_llm:utils",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:pynvml",
        "//rtp_llm:psutil",
        "//rtp_llm:fastapi",
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
    name = "backend_registry_test",
    srcs = [
        "backend_registry_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
    ],
)
