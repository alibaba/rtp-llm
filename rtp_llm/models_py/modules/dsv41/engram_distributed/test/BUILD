load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

package(default_visibility = ["//visibility:public"])

_ENGRAM = "//rtp_llm/models_py/modules/dsv41/engram_distributed"

_CPU_ENV = {"GPU_COUNT": "0", "CUDA_VISIBLE_DEVICES": ""}

_CUDA13 = select({
    "//:using_cuda13_arm": [],
    "//:using_cuda13_x86": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_test(
    name = "test_layout",
    size = "small",
    srcs = ["test_layout.py"],
    deps = [_ENGRAM + ":core"],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_storage",
    size = "small",
    srcs = ["test_storage.py", "test_layout.py"],
    deps = [_ENGRAM + ":core"],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

# Real world and cohort TCPStore clients use CPU Torch; allocation is injected.
py_test(
    name = "test_startup",
    size = "small",
    srcs = ["test_startup.py"],
    deps = [
        _ENGRAM + ":core",
        "//rtp_llm:torch",
    ],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_engram_model_attach",
    size = "small",
    srcs = ["test_engram_model_attach.py"],
    deps = [_ENGRAM + ":core"],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_resident_network",
    size = "small",
    srcs = ["test_resident_network.py", "test_startup.py"],
    deps = [_ENGRAM + ":core"],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_checkpoint_runtime",
    size = "small",
    srcs = ["test_checkpoint_runtime.py", "test_layout.py"],
    deps = [
        _ENGRAM + ":runtime",
        "//rtp_llm/models_py:engram_model_cpu",
    ],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
    target_compatible_with = _CUDA13,
)

py_test(
    name = "test_engram_model_loading",
    size = "small",
    srcs = ["test_engram_model_loading.py"],
    deps = [
        _ENGRAM + ":runtime",
        "//rtp_llm/models_py:engram_model_cpu",
        "//rtp_llm/models_py:engram_warmup_cpu",
        "//rtp_llm/models_py:engram_scale_source_cpu",
    ],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
    target_compatible_with = _CUDA13,
)

py_test(
    name = "test_embedding_contract",
    size = "small",
    srcs = ["test_embedding_contract.py"],
    deps = ["//rtp_llm/models_py:engram_model_cpu"],
    env = _CPU_ENV,
    legacy_create_init = 0,
    target_compatible_with = _CUDA13,
)

py_test(
    name = "test_fabric_ci",
    size = "small",
    srcs = ["run_fabric_ci.py", "test_fabric.py"],
    args = ["FabricGateTest"],
    main = "test_fabric.py",
    deps = [_ENGRAM + ":core"],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)

py_test(
    name = "test_native",
    size = "small",
    srcs = ["run_fabric_ci.py", "test_fabric.py"],
    main = "test_fabric.py",
    data = [
        "//rtp_llm/libs:fabric_memory_libs",
        "//rtp_llm/libs:libth_transformer_config_so",
    ],
    deps = [
        _ENGRAM + ":core",
        "//rtp_llm:ops",
        "//rtp_llm:torch",
    ],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
    target_compatible_with = _CUDA13,
)

# The default test wrapper hides CUDA (GPU_COUNT=0). Explicit invocations
# clear run_under so the runner can check and lock the requested idle GPUs.
py_test(
    name = "test_fabric_integration",
    size = "large",
    srcs = ["run_fabric_ci.py", "test_fabric.py"],
    main = "run_fabric_ci.py",
    data = [
        "//rtp_llm/libs:fabric_memory_libs",
        "//rtp_llm/libs:libth_transformer_config_so",
    ],
    deps = [
        _ENGRAM + ":runtime",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
    env = {"GPU_COUNT": "0"},
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "2"},
    tags = ["manual", "engram_fabric"],
    legacy_create_init = 0,
    target_compatible_with = select({
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

# Keep four-process POSIX_FD coverage on both CUDA13 SM100 hosts and x86 SM103.
cuda13_sm100_py_test(
    name = "test_gpu_storage",
    size = "large",
    timeout = "long",
    srcs = ["test_gpu_storage.py"],
    main = "test_gpu_storage.py",
    gpu_count = 4,
    data = ["//rtp_llm/libs:fabric_memory_libs"],
    deps = [
        _ENGRAM + ":runtime",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["engram_gpu_storage"],
    legacy_create_init = 0,
)

py_test(
    name = "test_gpu_storage_config",
    size = "small",
    srcs = ["test_gpu_storage.py"],
    main = "test_gpu_storage.py",
    args = ["LocalCudaEngramConfigTest"],
    deps = [_ENGRAM + ":core"],
    env = _CPU_ENV,
    tags = ["engram_distributed_cpu"],
    legacy_create_init = 0,
)
