package(default_visibility = ["//visibility:public"])

[
    sh_test(
        name = "local_prefill_sg_align_" + mode + "_test",
        srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
        args = ["local-prefill-sg-align-" + mode],
        data = [
            "test_local_prefill_sg_ops.py",
            "test_local_prefill_sg_topk.py",
            "test_local_prefill_sg_cache_rows.py",
            "test_local_prefill_sg_integration.py",
            "test_local_prefill_sg_paged_indexer.py",
            "test_local_prefill_sg_rope_lookup.py",
            "test_local_prefill_sg_rope_callers.py",
            ":local_prefill_workspace_contracts_test",
        ],
        env = {"GPU_COUNT": str(gpus)},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "long",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for mode, gpus in [("contracts", 0), ("device", 1)]
]

[
    sh_test(
        name = "local_prefill_workspace_" + mode + "_test",
        srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
        args = ["local-prefill-workspace-" + mode],
        data = [
            "test_local_prefill_workspace.py",
            "test_local_prefill_rows.py",
            "test_local_prefill_fused.py",
            "test_local_prefill_mhc.py",
            "test_local_prefill_aiu_mhc.py",
            "test_local_prefill_operator_flags.py",
            "test_fused_ops.py",
            "test_local_prefill_moe_routes.py",
            "test_local_prefill_compact_rows.py",
            "test_local_prefill_split_l20.py",
            "test_local_prefill_source_cache.py",
            "test_local_prefill_qo_rope.py",
            "test_dp_prefill_mega_capacity.py",
            "test_local_prefill_swa_segments.py",
            "test_local_prefill_token_topk.py",
            "test_local_prefill_token_topk_routes.py",
            "test_local_prefill_swa_native_write.py",
            "test_local_prefill_swa_pack.py",
            "test_local_prefill_engram_rounds.py",
            "test_local_prefill_rotary_pack.py",
            "test_local_prefill_engram_gate.py",
            "test_local_prefill_cross_layer_mhc.py",
            "test_local_prefill_shared_swiglu.py",
            "test_local_prefill_swa_native_read.py",
            "test_local_prefill_source_native_read.py",
            "test_local_prefill_source_native_write.py",
            "test_local_prefill_csa_native_write.py",
            "test_local_prefill_lazy_slots.py",
            "test_local_prefill_attention_provider.py",
            "//rtp_llm/models_py/modules/dsv41/fp8/test:test_v41_prefill_topk.py",
            "test_mhc_mega_probe.py",
            ":ptpc_sources",
            "//rtp_llm/models_py:v41_ppu_shared_sources",
            "//rtp_llm/tools/v41:tooling",
        ],
        env = {"GPU_COUNT": str(gpus)},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "long",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for mode, gpus in [("contracts", 0), ("device", 1)]
]

[
    sh_test(
        name = "local_prefill_attention_provider_" + mode + "_test",
        srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
        args = ["local-prefill-attention-provider-" + mode],
        # Reuse the declared fixture closure without rerunning the old suite.
        data = [":local_prefill_workspace_contracts_test"],
        env = {"GPU_COUNT": str(gpus)},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "long",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for mode, gpus in [("contracts", 0), ("device", 1)]
]

[
    sh_test(
        name = "local_prefill_token_topk_variants_" + mode + "_test",
        srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
        args = ["local-prefill-token-topk-variants-" + mode],
        data = [
            "test_local_prefill_token_topk_variants.py",
            "test_local_prefill_token_topk.py",
            "test_local_prefill_operator_flags.py",
            "//rtp_llm/models_py/modules/dsv41/fp8/test:test_v41_prefill_topk.py",
            "//rtp_llm/models_py:v41_ppu_shared_sources",
            "//rtp_llm/tools/v41:tooling",
        ],
        env = {"GPU_COUNT": str(gpus)},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "long",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for mode, gpus in [("contracts", 0), ("device", 1)]
]

[
    sh_test(
        name = "local_prefill_native_fp32_topk_variants_" + mode + "_test",
        srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
        args = ["local-prefill-native-fp32-topk-variants-" + mode],
        data = [
            "test_local_prefill_native_fp32_topk_variants.py",
            ":local_prefill_token_topk_variants_contracts_test",
        ],
        env = {"GPU_COUNT": str(gpus)},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "long",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for mode, gpus in [("contracts", 0), ("device", 1)]
]

[
    sh_test(
        name = "local_prefill_hierarchical_topk_variants_" + mode + "_test",
        srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
        args = ["local-prefill-hierarchical-topk-variants-" + mode],
        data = [
            "test_local_prefill_hierarchical_topk_variants.py",
            ":local_prefill_token_topk_variants_contracts_test",
        ],
        env = {"GPU_COUNT": str(gpus)},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "long",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for mode, gpus in [("contracts", 0), ("device", 1)]
]

[
    sh_test(
        name = "local_prefill_fused_hierarchical_topk_" + mode + "_test",
        srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
        args = ["local-prefill-fused-hierarchical-topk-" + mode],
        data = [
            "test_local_prefill_fused_hierarchical_topk.py",
            ":local_prefill_hierarchical_topk_variants_contracts_test",
        ],
        env = {"GPU_COUNT": str(gpus)},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "long",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for mode, gpus in [("contracts", 0), ("device", 1)]
]

sh_test(
    name = "dp_prefill_mega_capacity_distributed_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["dp-prefill-mega-capacity-distributed"],
    data = [
        "test_dp_prefill_mega_capacity.py",
        "test_local_prefill_moe_routes.py",
        "test_local_prefill_rows.py",
        "//rtp_llm/models_py:v41_ppu_shared_sources",
        "//rtp_llm/tools/v41:tooling",
    ],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "local_prefill_engram_rounds_distributed_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["local-prefill-engram-rounds-distributed"],
    data = [
        "test_local_prefill_engram_rounds.py",
        "//rtp_llm/models_py:v41_ppu_shared_sources",
        "//rtp_llm/tools/v41:tooling",
    ],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

filegroup(
    name = "ptpc_sources",
    srcs = [
        "test_ptpc.py",
        "test_ptpc_quant.py",
        "test_ptpc_norm.py",
        "test_ptpc_norm_quant.py",
        "test_ptpc_prefill_out.py",
        "test_ptpc_prefill_reuse.py",
        "test_ptpc_woa_direct.py",
        "test_ptpc_prepared.py",
        "test_ptpc_input_reuse.py",
        "test_ptpc_qnorm_reuse.py",
        "ptpc_microbench.py",
    ],
)

filegroup(
    name = "shared_eplb_native_sources",
    srcs = [
        "service_fixture.py",
        "dspark_workload.py",
        "ced_rdma_suite.py",
        "dspark_rdma_contract.py",
        "eplb_graph_evidence.py",
        "eplb_rdma_role.py",
        "dspark_graph_batch.py",
        "pd_rdma_role.py",
        "test_dspark_evidence.py",
    ],
)

py_test(
    name = "test_shared_eplb_contract",
    srcs = ["test_eplb.py"],
    main = "test_eplb.py",
    data = ["//rtp_llm/models_py:v41_ppu_shared_sources", "//rtp_llm:v41_runtime_contract", "//rtp_llm/tools/v41:tooling"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "cpu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

filegroup(
    name = "shared_dspark_modes_native_sources",
    srcs = [
        "service_fixture.py",
        "dspark_workload.py",
        "dspark_rdma_contract.py",
        "dspark_rdma_role.py",
        "dspark_graph_batch.py",
        "pd_rdma_role.py",
        "test_dspark_evidence.py",
    ],
)

sh_test(
    name = "shared_dspark_modes_boundary_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["shared-dspark-modes-boundary"],
    data = ["test_dspark_modes.py", "//rtp_llm/models_py:v41_ppu_shared_sources"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

filegroup(
    name = "shared_sharded_native_sources",
    srcs = [
        "service_fixture.py",
        "dspark_workload.py",
        "ced_rdma_suite.py",
        "dspark_rdma_contract.py",
        "rdma_perf.py",
        "pd_rdma_role.py",
        "test_dspark_evidence.py",
    ],
)

sh_test(
    name = "shared_sharded_boundary_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["shared-sharded-boundary"],
    data = ["test_cp_sharding.py", "//rtp_llm/models_py:v41_ppu_shared_sources"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

filegroup(
    name = "shared_optimized_native_sources",
    srcs = ["optimized_smoke_role.py"],
)

sh_test(
    name = "shared_optimized_boundary_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["shared-optimized-boundary"],
    data = [
        "test_packed_attention.py",
        "test_fused_ops.py",
        "test_projection_streams.py",
        "//rtp_llm/models_py:v41_ppu_shared_sources",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

filegroup(
    name = "shared_native_sources",
    srcs = ["pd_rdma_role.py", "service_fixture.py", "dspark_rdma_role.py"],
)

filegroup(
    name = "shared_graph_native_sources",
    srcs = ["graph_fidelity_role.py", "pd_rdma_role.py", "service_fixture.py"],
)

filegroup(
    name = "shared_graph_modes_native_sources",
    srcs = ["graph_modes_role.py", "pd_rdma_role.py", "service_fixture.py"],
)

filegroup(
    name = "shared_dspark_direct_native_sources",
    srcs = [":shared_graph_modes_native_sources", "//rtp_llm/models_py:v41_ppu_shared_sources"],
)

py_test(
    name = "test_shared_graph_contract",
    srcs = ["test_decode_graph.py"],
    main = "test_decode_graph.py",
    data = ["//rtp_llm/models_py:v41_ppu_shared_sources"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "cpu"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "shared_prefill_adapter_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["shared-prefill-adapter"],
    data = ["test_prefill_runtime.py", "//rtp_llm/models_py:v41_ppu_shared_sources"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "shared_model_smoke_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["shared-model"],
    data = [
        "test_model.py",
        "//rtp_llm/models_py:v41_ppu_shared_sources",
    ],
    env = {
        "GPU_COUNT": "1",
        "RTP_V41_PPU_MOE_BACKEND": "grouped",
        "RTP_V41_PPU_MOE_PRECISION": "w4a8",
    },
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "mhc_mega_integration_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["mhc-integration"],
    env = {"GPU_COUNT": "1", "RTP_V41_TEST_MHC_DEVICE": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "dspark_contract_test",
    srcs = ["run_dspark_contracts.sh"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "cpu"],
    timeout = "long",
)

sh_test(
    name = "ced_contract_test",
    srcs = ["run_ced_contracts.sh"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "cpu"],
    timeout = "long",
)

py_test(
    name = "test_moe_megamoe",
    srcs = ["test_moe_megamoe.py"],
    deps = ["//rtp_llm/models_py:v41_ppu_moe"],
    env = {"GPU_COUNT": "0"},
    tags = ["cpu"],
)

py_test(
    name = "test_moe_w4a8",
    srcs = ["test_moe_w4a8.py"],
    deps = ["//rtp_llm/models_py:v41_ppu_moe"],
    env = {
        "GPU_COUNT": "0",
        "RTP_V41_TEST_W4A8_DEVICE": "0",
    },
    tags = ["cpu"],
)

sh_test(
    name = "test_moe_w4a8_device",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["moe-device"],
    env = {
        "GPU_COUNT": "1",
        "RTP_V41_TEST_W4A8_DEVICE": "1",
    },
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "s5_kernel_surface_probe_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["s5-kernel-surface"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "s5_mhc_mega_probe_test",
    srcs = ["//example/deepseek_v41/ppu:test_functional.sh"],
    args = ["s5-mhc-mega"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
