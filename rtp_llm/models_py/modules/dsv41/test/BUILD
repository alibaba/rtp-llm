load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")
load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")

package(default_visibility = ["//visibility:public"])

_TESTS = [
    "mega_moe_static_buf_test",
    "test_dspark_cuda_graph_contract",
    "test_engram_cpu",
    "test_engram_cpu_deadline",
    "test_engram_cpu_native_packed",
    "test_engram_packed",
    "test_engram_prefetch",
    "test_gate_routing_metadata",
    "test_hc_expand",
    "test_mega_gate",
    "test_mega_input_pack",
    "test_mega_se_input_pack",
    "test_mhc_lazy_backward",
    "test_prefill_workspace",
    "test_strategy_select",
    "test_silu_mul_bf16",

    "test_native_prefill_inputs",
    "test_mxfp8_dense",
    "test_deep_gemm_scratch",
    "test_cache_specs",
    "test_cache_tags",
    "test_ced",
    "test_dsv41_kernel_jit_warmup",
    "test_engram",
    "test_engram_cuda",
    "test_flash_mla_heads",
    "test_v41_cross_layer_mhc",
    "test_v41_decode_rope_metadata",
    "test_v41_delayed_hc",
    "test_v41_delayed_hc_cuda_graph",
    "test_v41_fused_inv_rope_fp8_quant",
    "test_v41_fused_qkv",
    "test_v41_hc_jit_warmup",
    "test_v41_mega_mhc",
    "test_v41_mega_mhc_compact",
    "test_v41_mhc_prenorm",
    "test_v41_query_quant",
]

[
    py_test(
        name = name,
        srcs = glob(["*.py"]) + ["//rtp_llm/test/pytest:pytest_main.py"],
        main = "//rtp_llm/test/pytest:pytest_main.py",
        args = ["--cuda-devices", "1", "$(location :" + name + ".py)"],
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:modules",
            "//rtp_llm/models_py:tile_kernels_mhc",
            "//rtp_llm:jit_cache_env",
            test_requirement("pytest"),
        ],
        tags = ["dsv41_cuda13", "SM100_ARM_CU13"],
        exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
        target_compatible_with = select({
            "@//:using_cuda13_arm": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        legacy_create_init = 0,
    )
    for name in _TESTS if name not in ["test_v41_mega_mhc"]
]

py_test(
    name = "test_embedding_tp4",
    srcs = ["test_embedding_tp4.py"],
    main = "test_embedding_tp4.py",
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
    tags = ["dsv41_cuda13", "SM100_ARM_CU13"],
    env = {"GPU_COUNT": "4"},
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "4"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

[
    cuda13_sm100_py_test(
        name = name,
        srcs = glob(["*.py"]) + ["//rtp_llm/test/pytest:pytest_main.py"],
        main = "//rtp_llm/test/pytest:pytest_main.py",
        args = ["--cuda-devices", "1", "$(location :" + name + ".py)"],
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:modules",
            "//rtp_llm/models_py:tile_kernels_mhc",
            "//rtp_llm:jit_cache_env",
            test_requirement("pytest"),
        ],
        tags = ["dsv41_cuda13"],
        legacy_create_init = 0,
    )
    for name in ["test_v41_mega_mhc"]
]

# This metadata test needs only Python's standard library, on any CPU.
py_test(
    name = "test_engram_manifest",
    srcs = ["test_engram_manifest.py"],
    main = "test_engram_manifest.py",
    deps = ["//rtp_llm/models_py:engram_manifest"],
    legacy_create_init = 0,
)

# Standalone unittest entry points: the real modules are declared as deps,
# while native allocation / CUDA graph behavior is injected by the CPU tests.
# Do not route these through pytest_main, testlib or the GPU _TESTS loop.
py_test(
    name = "test_engram_storage",
    size = "small",
    srcs = ["test_engram_storage.py"],
    main = "test_engram_storage.py",
    deps = [
        "//rtp_llm/models_py:engram_manifest",
        "//rtp_llm/models_py:engram_storage_cpu",
        "//rtp_llm/models_py:engram_model_attach_cpu",
    ],
    legacy_create_init = 0,
)

# Capacity policy and arithmetic must run without torch or CUDA.
py_test(
    name = "test_engram_capacity",
    size = "small",
    env = {"GPU_COUNT": "0"},
    srcs = ["test_engram_capacity.py"],
    main = "test_engram_capacity.py",
    deps = ["//rtp_llm/models_py:engram_capacity"],
    legacy_create_init = 0,
)

# This test loads the lightweight helper by file path and injects every model
# factory. It must run without torch, CUDA, native libraries or the RTP engine.
py_test(
    name = "test_engram_model_attach",
    size = "small",
    env = {"GPU_COUNT": "0"},
    srcs = ["test_engram_model_attach.py"],
    main = "test_engram_model_attach.py",
    deps = ["//rtp_llm/models_py:engram_model_attach_cpu"],
    legacy_create_init = 0,
)

# The test imports test_engram_manifest.py as its standard document fixture;
# declaring it explicitly is required for sandboxed runfiles. Real CPU hash
# constants come from the narrow engram_model_cpu closure, never :modules.
py_test(
    name = "test_engram_bootstrap",
    size = "small",
    env = {"GPU_COUNT": "0"},
    srcs = ["test_engram_bootstrap.py", "test_engram_manifest.py"],
    main = "test_engram_bootstrap.py",
    deps = ["//rtp_llm/models_py:engram_bootstrap_cpu"],
    legacy_create_init = 0,
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

# CPU execution with the validated CUDA13 ARM dependency versions. These
# tests need modern FP8/E8M0 torch/safetensors, but no GPU allocation or runtime.
[
    py_test(
        name = name,
        size = "small",
        env = {"GPU_COUNT": "0"} if name == "test_engram_checkpoint" else {},
        srcs = [name + ".py"],
        main = name + ".py",
        deps = ["//rtp_llm/models_py:" + library],
        legacy_create_init = 0,
        target_compatible_with = select({
            "@//:using_cuda13_arm": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, library in [
        ("test_engram_checkpoint", "engram_checkpoint_cpu"),
        ("test_engram_external_checkpoint", "engram_model_cpu"),
        ("test_engram_external_warmup", "engram_warmup_cpu"),
    ]
]

# Distributed layout and rollback tests do not import Torch or CUDA.
[
    py_test(
        name = name,
        size = "small",
        srcs = [name + ".py"],
        main = name + ".py",
        deps = ["//rtp_llm/models_py:engram_storage_cpu"],
        legacy_create_init = 0,
    )
    for name in ["test_engram_shard", "test_engram_shard_config", "test_engram_shard_storage"]
]

py_test(
    name = "test_engram_exchange",
    size = "small",
    srcs = ["test_engram_exchange.py"],
    main = "test_engram_exchange.py",
    deps = [
        "//rtp_llm/models_py:engram_exchange_cpu",
        "//tools/engram_sidecar/framework:exchange_test_certs",
    ],
    # Uses localhost TLS sockets and the test host's openssl command.
    tags = ["requires-network", "requires-openssl"],
    legacy_create_init = 0,
)
