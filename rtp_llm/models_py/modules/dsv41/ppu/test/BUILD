package(default_visibility = ["//visibility:public"])

filegroup(
    name = "ptpc_sources",
    srcs = ["test_ptpc.py"],
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
