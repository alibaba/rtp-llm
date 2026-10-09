# DSV4 python tests target the CUDA13/Blackwell stack (deep_gemm >= 2.6, dsv4
# kernels, pytest closure). They are skipped as incompatible on other platforms
# instead of failing on pools that can never satisfy those dependencies.
_DSV4_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

_DSV4_SM120_ONLY = select({
    "@//:using_cuda13_sm120": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

_DSV4_TEST_DEPS = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
    "//rtp_llm:testlib",
]

py_test(
    name = "test_cp_context_build",
    srcs = ["test_cp_context_build.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_topk_idxs_align",
    srcs = ["test_cp_topk_idxs_align.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_pool_block_gather",
    srcs = ["test_cp_pool_block_gather.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_pool_reader",
    srcs = ["test_pool_reader.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_kv_allgather_restore",
    srcs = ["test_cp_kv_allgather_restore.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_slot_mapping",
    srcs = ["test_cp_slot_mapping.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_indexer_cp_assembler",
    srcs = ["test_indexer_cp_assembler.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_via_concat_meta",
    srcs = ["test_cp_via_concat_meta.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_indexer_seq_total",
    srcs = ["test_cp_indexer_seq_total.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_workspace_combine_topk",
    srcs = ["test_cp_workspace_combine_topk.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_mtp_hidden_buffer",
    srcs = ["test_mtp_hidden_buffer.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_dspark_cuda_graph_contract",
    srcs = ["test_dspark_cuda_graph_contract.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_dspark_runtime_config",
    srcs = ["test_dspark_runtime_config.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_aux_hidden_capture",
    srcs = ["test_aux_hidden_capture.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_attention_merge",
    srcs = ["test_cp_attention_merge.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_attention_shard",
    srcs = ["test_cp_attention_shard.py"],
    deps = _DSV4_TEST_DEPS,
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_library(
    name = "dsv4_test_utils",
    srcs = ["dsv4_test_utils.py"],
    deps = _DSV4_TEST_DEPS,
)

py_library(
    name = "dsv4_kernel_perf_utils",
    srcs = ["dsv4_kernel_perf_utils.py"],
)

py_test(
    name = "test_cp_async",
    srcs = ["test_cp_async.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_prefill_workspace",
    srcs = ["test_prefill_workspace.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_prefill_fast_path",
    srcs = ["test_prefill_fast_path.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_hc_impl",
    srcs = ["test_hc_impl.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:tile_kernels_mhc",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_strategy_select",
    srcs = ["test_strategy_select.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_chunked_moe",
    srcs = ["test_chunked_moe.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_moe_communication",
    srcs = ["test_moe_communication.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_sm120_collective_hardware",
    srcs = ["test_sm120_collective_hardware.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["RTX_5000_PRO"],
    exec_properties = {
        "gpu": "RTX_5000_PRO",
        "gpu_count": "2",
    },
    target_compatible_with = _DSV4_SM120_ONLY,
)

py_test(
    name = "test_sm120_fused_moe_capacity",
    srcs = ["test_sm120_fused_moe_capacity.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_sm120_a2a_payload_guard",
    srcs = ["test_sm120_a2a_payload_guard.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_indexer_topk_backend",
    srcs = ["test_indexer_topk_backend.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_shared_expert_executor",
    srcs = ["test_shared_expert_executor.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_sm120_small_batch_projections",
    srcs = ["test_sm120_small_batch_projections.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["RTX_5000_PRO"],
    exec_properties = {"gpu": "RTX_5000_PRO", "gpu_count": "1"},
    target_compatible_with = _DSV4_SM120_ONLY,
)

py_test(
    name = "test_mega_moe_input_packer",
    srcs = ["test_mega_moe_input_packer.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_mega_moe_se_input_pack",
    srcs = ["test_mega_moe_se_input_pack.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_mega_moe_gate_pack",
    srcs = ["test_mega_moe_gate_pack.py"],
    deps = ["//rtp_llm/models_py:dsv4_fused_kernel_sources"],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_dsv4_kernel_jit_warmup",
    srcs = ["test_dsv4_kernel_jit_warmup.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_mega_moe_input_packer_perf",
    srcs = ["test_mega_moe_input_packer_perf.py"],
    deps = ["//rtp_llm/models_py:dsv4_fused_kernel_sources"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_grouped_fp4_strategy_perf",
    srcs = ["test_grouped_fp4_strategy_perf.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_fused_inv_rope_fp8_quant",
    srcs = ["test_fused_inv_rope_fp8_quant.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_fused_rmsnorm_rope",
    srcs = ["test_fused_rmsnorm_rope.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_fused_rmsnorm_fp8_quant",
    srcs = ["test_fused_rmsnorm_fp8_quant.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_rope_only",
    srcs = ["test_rope_only.py"],
    deps = [
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_fused_inv_rope_fp8_quant_perf",
    srcs = ["test_fused_inv_rope_fp8_quant_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_fused_rmsnorm_rope_perf",
    srcs = ["test_fused_rmsnorm_rope_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_rope_only_perf",
    srcs = ["test_rope_only_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_cp_restore_perf",
    srcs = ["test_cp_restore_perf.py"],
    deps = [
        ":dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_dsv4_tc_simt_overlap_perf",
    srcs = ["test_dsv4_tc_simt_overlap_perf.py"],
    deps = [
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "SM100_ARM"],
    exec_properties = {"gpu": "SM100_ARM", "gpu_count": "1"},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

_PREFILL_POLICY_SOURCE_DATA = [
    "//rtp_llm/cpp/normal_engine:prefill_pipeline_source_data",
    "//rtp_llm/cpp/models:prefill_wire_source_data",
    "//rtp_llm/models_py:prefill_policy_source_data",
    "//rtp_llm/models_py/distributed:rank_layout",
]

py_test(
    name = "test_prefill_integration_policy",
    srcs = ["test_prefill_integration_policy.py"],
    data = _PREFILL_POLICY_SOURCE_DATA,
)

py_test(
    name = "test_pd_transfer_geometry",
    srcs = ["test_pd_transfer_geometry.py"],
    data = ["//rtp_llm:dsv4_kv_cache_source"],
)

py_test(
    name = "test_prefill_scale_cpu",
    srcs = ["test_prefill_scale_cpu.py", "test_prefill_integration_policy.py"],
    main = "test_prefill_scale_cpu.py",
    data = _PREFILL_POLICY_SOURCE_DATA,
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "test_forward_ep_plan_cpu",
    srcs = ["test_forward_ep_plan_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = ["//rtp_llm/models_py:prefill_optimization_source_data"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_prefill_diagnostics_cpu",
    srcs = ["test_prefill_diagnostics_cpu.py"],
    data = ["//rtp_llm/models_py:prefill_optimization_source_data"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_moe_final_out_cache_cpu",
    srcs = ["test_moe_final_out_cache_cpu.py", "moe_cache_test_support.py"],
    main = "test_moe_final_out_cache_cpu.py",
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm:testlib"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
    target_compatible_with = _DSV4_CUDA13_ONLY,
)

py_test(
    name = "test_moe_final_out_cache_cuda",
    srcs = ["test_moe_final_out_cache_cuda.py", "moe_cache_test_support.py"],
    main = "test_moe_final_out_cache_cuda.py",
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO"],
    exec_properties = {"gpu": "RTX_5000_PRO", "gpu_count": "1"},
    target_compatible_with = _DSV4_SM120_ONLY,
)

# Pure CPU boundary checks against actual native config and production helpers.
py_test(
    name = "test_cep_dspark_prefill",
    srcs = ["test_cep_dspark_prefill.py"],
    args = ["CEPDSparkPolicyTest"],
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm:testlib"],
    target_compatible_with = _DSV4_SM120_ONLY,
    env = {"CUDA_VISIBLE_DEVICES": "", "GPU_COUNT": "0"},
)

# Execution-model imports select FusedRope by real device capability. Do not
# force a CPU fallback in this CUDA-only build or monkeypatch device detection.
py_test(
    name = "test_cep_dspark_prefill_model",
    srcs = ["test_cep_dspark_prefill.py"],
    main = "test_cep_dspark_prefill.py",
    args = ["CEPDSparkModelTest"],
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm:testlib"],
    target_compatible_with = _DSV4_SM120_ONLY,
    env = {"GPU_COUNT": "1"},
    tags = ["manual"],
)

py_test(
    name = "test_pp_draft_weight_loading",
    srcs = ["test_pp_draft_weight_loading.py"],
    args = [
        "PPDraftWeightLoadingTest.test_policy_preserves_target_and_single_stage_loading",
        "PPDraftWeightLoadingTest.test_last_stage_loads_without_entering_other_stages_world_collectives",
    ],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _DSV4_SM120_ONLY,
    env = {"CUDA_VISIBLE_DEVICES": "", "GPU_COUNT": "0"},
    timeout = "short",
)

py_test(
    name = "test_pp_draft_weight_loading_cuda",
    srcs = ["test_pp_draft_weight_loading.py"],
    main = "test_pp_draft_weight_loading.py",
    args = ["PPDraftWeightLoadingTest.test_cuda_local_reader_owns_values_after_close"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _DSV4_SM120_ONLY,
    env = {"GPU_COUNT": "1"},
    tags = ["manual"],
)

py_test(
    name = "test_cp_context_plan_cache_cpu",
    srcs = ["test_cp_context_plan_cache_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_cp_deferred_verify_cpu",
    srcs = ["test_cp_deferred_verify_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_cp_host_staging_cpu",
    srcs = ["test_cp_host_staging_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_head_host_mirror_scalars_cpu",
    srcs = ["test_head_host_mirror_scalars_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_host_run_ahead_counts_cpu",
    srcs = ["test_host_run_ahead_counts_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_prefill_async_head_cpu",
    srcs = ["test_prefill_async_head_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_prefill_meta_shared_build_cpu",
    srcs = ["test_prefill_meta_shared_build_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_suffix_gather_max_host_cpu",
    srcs = ["test_suffix_gather_max_host_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_mhc_prenorm_dg_cpu",
    srcs = ["test_mhc_prenorm_dg_cpu.py"],
    deps = ["//rtp_llm:torch"],
    data = [
        "//rtp_llm/models_py:prefill_optimization_source_data",
        ":cpu_test_support",
        "//rtp_llm:device_type_source",
    ],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

filegroup(
    name = "cpu_test_support",
    srcs = ["cpu_test_utils.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_cpu_test_utils",
    srcs = ["test_cpu_test_utils.py"],
    deps = ["//rtp_llm:torch"],
    data = [":cpu_test_support"],
    env = {"CUDA_VISIBLE_DEVICES": ""},
)
