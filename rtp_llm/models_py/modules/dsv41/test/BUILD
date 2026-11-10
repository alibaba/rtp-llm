load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")

package(default_visibility = ["//visibility:public"])

# CPU-only contract tests: no CUDA context, no GPU launches (docstrings and
# code audited; CUDA references are mocks only). They drop the SM100 GPU pin
# and run on the ARM CPU pool via the cuda13_arm default exec properties, with
# GPU_COUNT=0 so the device_resource wrapper skips CUDA probes and GPU locks.
# The SM100_ARM_CU13 tag is kept on purpose: lane selection uses a positive
# tag filter, while worker routing comes from exec_properties only.
_CPU_ONLY_TESTS = [
    "test_cache_specs",
    "test_cache_tags",
    "test_deep_gemm_scratch",
    "test_engram",
    "test_flash_mla_heads",
    "test_v41_delayed_hc",
    "test_v41_mega_mhc_compact",
]

_TESTS = [
    "test_native_prefill_inputs",
    "test_mxfp8_dense",
    "test_ced",
    "test_dsv41_kernel_jit_warmup",
    "test_engram_cuda",
    "test_v41_cross_layer_mhc",
    "test_v41_decode_rope_metadata",
    "test_v41_delayed_hc_cuda_graph",
    "test_v41_fused_inv_rope_fp8_quant",
    "test_v41_fused_qkv",
    "test_v41_hc_jit_warmup",
    "test_v41_mega_mhc",
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
    for name in _TESTS
]

[
    py_test(
        name = name,
        srcs = glob(["*.py"]) + ["//rtp_llm/test/pytest:pytest_main.py"],
        main = "//rtp_llm/test/pytest:pytest_main.py",
        args = ["$(location :" + name + ".py)"],
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:modules",
            "//rtp_llm/models_py:tile_kernels_mhc",
            "//rtp_llm:jit_cache_env",
            test_requirement("pytest"),
        ],
        env = {"GPU_COUNT": "0"},
        tags = ["dsv41_cuda13", "SM100_ARM_CU13"],
        target_compatible_with = select({
            "@//:using_cuda13_arm": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        legacy_create_init = 0,
    )
    for name in _CPU_ONLY_TESTS
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
