load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps", "triton_deps")

cc_library(
    name = "mtp_policy",
    srcs = ["MtpPolicy.cc"],
    hdrs = ["MtpPolicy.h", "ppu/MtpPolicy.h"],
    copts = copts(),
    deps = ["//rtp_llm/cpp/cache"] + torch_deps(),
    visibility = ["//visibility:public"],
)

cc_test_wrapper(
    name = "mtp_policy_test",
    srcs = ["ppu/test/MtpPolicyTest.cc"],
    copts = copts(),
    deps = [
        ":mtp_policy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    exec_properties = select({
        "@//:using_ppu": {"gpu": "PPU-ZW810E"},
        "//conditions:default": {"gpu": "L20_CU13"},
    }),
    tags = ["gpu"],
)

_PPU_CONTRACT_TESTS = [
    "mtp_weights", "mtp_hidden", "compressor_norm_dispatch", "mtp_swa_ring_page",
    "chunked_moe", "deepep_graph_dispatch", "ppu_deep_gemm_layout_compat",
    "strategy_select", "tp_norm", "v4_router_contract", "v4_tp_reduce_contract",
]

# Model contracts use the PPU dependency closure and test workers.
[py_test(
    name = name,
    srcs = ["ppu/test/test_" + name + ".py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_" + name + ".py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    timeout = "moderate",
) for name in _PPU_CONTRACT_TESTS]

# PPU operators and JIT sources are packaged only for the PPU build.
[py_test(
    name = name,
    srcs = ["ppu/test/test_" + name + ".py"],
    main = "ppu/test/test_" + name + ".py",
    data = [":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
) for name in [
    "isolated_metadata", "rich_contract", "chunk_options", "platform_provider",
    "prefill_communication", "tp_prefill_workspace_dims",
]]

_ENTRYPOINTS = ["__init__.py", "model_dispatch.py", "ppu/__init__.py"]

py_library(
    name = "ppu_test_runtime",
    deps = ["//rtp_llm:config", "//rtp_llm:testlib", "//rtp_llm/models_py:models"] + select({
        "@//:using_ppu": ["//rtp_llm/models_py:deep_gemm", "//rtp_llm/models_py:deep_ep", "//rtp_llm:tilelang"],
        "//conditions:default": [],
    }) + triton_deps(["triton"]),
)

py_test(
    name = "sglang_jit_test",
    srcs = ["ppu/test/test_sglang_jit.py"],
    main = "ppu/test/test_sglang_jit.py",
    data = [":ppu_sources"],
)

[py_test(
    name = name + "_test",
    srcs = ["ppu/test/test_" + name + ".py"],
    main = "ppu/test/test_" + name + ".py",
    data = [":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
    tags = ["manual", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
) for name in ["mega_moe_contract", "mtp_input", "decode_metadata_graph", "decode_state_slots"]]

py_test(
    name = "decode_fusions_test",
    srcs = ["ppu/test/test_decode_fusions.py"],
    main = "ppu/test/test_decode_fusions.py",
    data = [":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
    tags = ["manual", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "platform_boundary_test",
    srcs = ["ppu/test/test_platform_boundary.py"],
    main = "ppu/test/test_platform_boundary.py",
    data = ["model_dispatch.py", "ppu/models/dsv4/resources.py"],
)

py_test(
    name = "ep_chunk_rounds_test",
    srcs = ["ppu/test/test_ep_chunk_rounds.py"],
    main = "ppu/test/test_ep_chunk_rounds.py",
    data = ["ppu/models/dsv4/ppu_ep_moe.py"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "indexer_cache_test",
    srcs = ["ppu/test/test_indexer_cache.py"],
    main = "ppu/test/test_indexer_cache.py",
    data = ["ppu/models/dsv4/cache.py"],
)

py_test(
    name = "dsv4_config_test",
    srcs = ["ppu/test/test_dsv4_config.py"],
    main = "ppu/test/test_dsv4_config.py",
    data = ["ppu/models/dsv4/manifest.py"],
)

filegroup(
    name = "ppu_sources",
    srcs = glob(["ppu/**/*.py", "ppu/**/*.cuh", "ppu/**/*.h", "ppu/**/*.json", "ppu/**/*.patch", "ppu/**/NOTICE*"], exclude = ["**/test/**", "**/tests/**"] + _ENTRYPOINTS),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "platform_sources",
    srcs = _ENTRYPOINTS + select({
        "@//:using_ppu": [":ppu_sources"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_test(
    name = "fp8_contract_test",
    srcs = ["ppu/test/test_fp8_contracts.py"],
    main = "ppu/test/test_fp8_contracts.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
)

py_test(
    name = "fused_permute_test",
    srcs = ["ppu/test/test_fused_permute.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_fused_permute.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm:torch"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "mxfp4_masked_test",
    srcs = ["ppu/test/test_mxfp4_masked.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_mxfp4_masked.py)"],
    env = {"GPU_COUNT": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "local", "gpu", "M890P"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "flash_mla_runtime_test",
    srcs = ["ppu/test/test_flash_mla_runtime.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_flash_mla_runtime.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm/models_py:flash-mla", "//rtp_llm:torch"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_hc_test",
    srcs = ["ppu/test/test_decode_hc.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_decode_hc.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "mxfp4_deepep_test",
    srcs = ["ppu/test/test_mxfp4_deepep.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "4", "$(location ppu/test/test_mxfp4_deepep.py)"],
    env = {"GPU_COUNT": "4"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "local", "gpu", "M890P"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_attention_test",
    srcs = ["ppu/test/test_decode_attention.py"],
    main = "ppu/test/test_decode_attention.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "decode_shared_expert_test",
    srcs = ["ppu/test/test_decode_shared_expert.py"],
    main = "ppu/test/test_decode_shared_expert.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "fp4_indexer_request_slices_test",
    srcs = ["ppu/test/test_fp4_indexer_request_slices.py"],
    main = "ppu/test/test_fp4_indexer_request_slices.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

test_suite(
    name = "ppu_source_regression_tests",
    tests = [
        ":fp4_indexer_request_slices_test",
    ],
)

_COMMON_CONTRACT_TESTS = [
    ":sglang_jit_test",
    ":platform_boundary_test", ":dsv4_config_test", ":indexer_cache_test",
    ":ep_chunk_rounds_test", ":isolated_metadata", ":rich_contract",
    ":chunk_options", ":platform_provider", ":prefill_communication",
    ":tp_prefill_workspace_dims", ":fp8_contract_test", ":mtp_policy_test",
]

# These validate PPU imports and mocked contracts, not M890P kernel numerics.
test_suite(
    name = "dsv4_ppu_runtime_contracts",
    tests = _COMMON_CONTRACT_TESTS + [":" + name for name in _PPU_CONTRACT_TESTS] + [
        ":fp4_indexer_request_slices_test",
        ":decode_moe_hint_test", ":decode_shared_schedule_test",
        ":ppu_tp_shared_weights_test", ":ppu_hc_head_options_test",
        ":prefill_defaults_test",
    ],
)

py_test(
    name = "fp4_indexer_chunk_state_test",
    srcs = ["ppu/test/test_fp4_indexer_chunk_state.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_fp4_indexer_chunk_state.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "bf16_radix_topk_test",
    srcs = ["ppu/test/test_bf16_radix_topk.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_bf16_radix_topk.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "mtp_queries_test",
    srcs = ["ppu/models/dsv4/test/test_ppu_mtp_queries.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/models/dsv4/test/test_ppu_mtp_queries.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_qkv_test",
    srcs = ["ppu/test/test_decode_qkv.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_decode_qkv.py)"],
    env = {"GPU_COUNT": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "local", "gpu", "M890P"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_hc_norm_test",
    srcs = ["ppu/test/test_decode_hc_norm.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_decode_hc_norm.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_fp8_quant_test",
    srcs = ["ppu/test/test_decode_fp8_quant.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_decode_fp8_quant.py)"],
    env = {"GPU_COUNT": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "local", "gpu", "M890P"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_indexer_overlap_test",
    srcs = ["ppu/test/test_decode_indexer_overlap.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_decode_indexer_overlap.py)"],
    env = {"GPU_COUNT": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "local", "gpu", "M890P"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_moe_hint_test",
    srcs = ["ppu/test/test_decode_moe_hint.py"],
    main = "ppu/test/test_decode_moe_hint.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_moe_output_test",
    srcs = ["ppu/test/test_decode_moe_output.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_decode_moe_output.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "decode_shared_schedule_test",
    srcs = ["ppu/test/test_decode_shared_schedule.py"],
    main = "ppu/test/test_decode_shared_schedule.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
    tags = ["gpu", "ppu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_tp_shared_weights_test",
    srcs = ["ppu/models/dsv4/test/test_ppu_tp_shared_weights.py"],
    main = "ppu/models/dsv4/test/test_ppu_tp_shared_weights.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "topk_padding_test",
    srcs = ["ppu/test/test_topk_padding.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_topk_padding.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_hc_head_options_test",
    srcs = ["ppu/models/dsv4/test/test_ppu_hc_head_options.py"],
    main = "ppu/models/dsv4/test/test_ppu_hc_head_options.py",
    data = [":platform_sources", ":ppu_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"] + select({"@//:using_ppu": [":ppu_test_runtime"], "//conditions:default": []}),
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "silu_mul_mxfp4_test",
    srcs = ["ppu/test/test_silu_mul_mxfp4.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_silu_mul_mxfp4.py)"],
    env = {"GPU_COUNT": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "local", "gpu", "M890P"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

# Self-contained operator checks; checkpoint-dependent model tests stay manual.
py_test(
    name = "prefill_defaults_test",
    srcs = ["ppu/test/test_prefill_defaults.py"],
    main = "ppu/test/test_prefill_defaults.py",
    data = [":ppu_sources"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "prefill_optimizations_test",
    srcs = ["ppu/test/test_prefill_optimizations.py", "ppu/test/ppu_test_main.py"],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_prefill_optimizations.py)"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["PPU-ZW810E"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

# SDK FP8/FP4 AIU kernels require PPU 1.5. Run explicitly on a local M890P;
# do not schedule them on the automatic PPU 1.0/810E worker pool.
py_test(
    name = "prefill_m890p_test",
    srcs = [
        "ppu/test/test_prefill_m890p.py",
        "ppu/test/test_prefill_optimizations.py",
        "ppu/test/ppu_test_main.py",
    ],
    main = "ppu/test/ppu_test_main.py",
    args = ["--devices", "1", "$(location ppu/test/test_prefill_m890p.py)"],
    env = {"GPU_COUNT": "1"},
    data = [":platform_sources", ":ppu_sources"],
    deps = [":ppu_test_runtime", "//rtp_llm:device", "//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "local", "gpu", "M890P"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

test_suite(
    name = "dsv4_ppu_m890p_manual_tests",
    tags = ["manual"],
    tests = [
        ":mxfp4_masked_test",
        ":mxfp4_deepep_test",
        ":decode_qkv_test",
        ":decode_fp8_quant_test",
        ":decode_indexer_overlap_test",
        ":silu_mul_mxfp4_test",
        ":prefill_m890p_test",
        ":decode_attention_test",
        ":decode_shared_expert_test",
        ":decode_fusions_test",
        ":decode_metadata_graph_test",
        ":decode_state_slots_test",
        ":mtp_input_test",
    ],
)

test_suite(
    name = "dsv4_ppu_numerical_tests",
    tests = [
        ":fused_permute_test",
        ":flash_mla_runtime_test",
        ":decode_hc_test",
        ":fp4_indexer_chunk_state_test",
        ":bf16_radix_topk_test",
        ":mtp_queries_test",
        ":decode_hc_norm_test",
        ":decode_moe_output_test",
        ":topk_padding_test",
        ":prefill_optimizations_test",
    ],
)
