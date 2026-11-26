load("@arch_config//:arch_select.bzl", "requirement", "triton_deps")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")
load("//rtp_llm/models_py/test:defs.bzl", "models_py_test")
load(":checkpoint_service_tests.bzl", "checkpoint_service_tests")
load(":checkpoint_test_platforms.bzl", "checkpoint_reader_test_env", "ue8m0_test_env", "ue8m0_test_platforms")

py_library(
    name = "checkpoint_reader_test_support",
    srcs = ["checkpoint_reader_test_support.py"],
    testonly = True,
    visibility = ["//rtp_llm:__subpackages__"],
)

py_test(
    name = "v4_fst_nogds_test",
    srcs = ["v4_fst_nogds_test.py"],
    deps = [":new_loader_core"],
)

py_test(
    name = "v4_fst_nogds_gpu_test",
    srcs = ["v4_fst_nogds_gpu_test.py"],
    deps = [":new_loader_core", "//rtp_llm:fastsafetensors"],
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "v41_fst_file_pool_test",
    srcs = ["v41_fst_file_pool_test.py"],
    deps = [":new_loader_core", "//rtp_llm:fastsafetensors"],
    env = {"FASTSAFETENSORS_NOGDS": "1"},
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "fst_host_backend_test",
    srcs = ["fst_host_backend_test.py"],
    deps = [":new_loader_core"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""},
)

py_test(
    name = "checkpoint_lifecycle_test",
    srcs = ["checkpoint_lifecycle_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""},
    deps = [":new_loader_core"],
)

py_test(
    name = "qwen4_media_embedding_test",
    srcs = ["new_models/qwen4_exp/test/test_media_embedding.py"],
    main = "new_models/qwen4_exp/test/test_media_embedding.py",
    env = checkpoint_reader_test_env({
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        ":new_loader_core",
        ":models",
        "//rtp_llm:models",
        "//rtp_llm:multimodal",
        "//rtp_llm:fastsafetensors",
    ],
)

# Qwen4 checkpoint owners and exact native service entrypoints. CPU execution
# validates loading/storage, not device inference or golden accuracy.
[
    py_test(
        name = "qwen4_direct_" + part + "_test",
        srcs = ["new_models/qwen4_exp/test/test_" + part + ".py"],
        main = "new_models/qwen4_exp/test/test_" + part + ".py",
        env = checkpoint_reader_test_env({
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }),
        deps = [":checkpoint_reader_test_support", ":new_weight_loader", ":new_loader_core", ":models", "//rtp_llm:fastsafetensors"],
    )
    for part in [
        "attention_weights", "checkpoint_validation", "dense_profiles",
        "gdn_weights", "hyper_connection", "ple_layer", "ple_weights",
        "qsa_indexer", "target_weights",
    ]
]

[
    py_test(
        name = name,
        srcs = [src],
        main = src,
        env = checkpoint_reader_test_env({
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }),
        deps = [":checkpoint_reader_test_support", ":new_weight_loader", ":new_loader_core", ":models", "//rtp_llm:fastsafetensors"],
    )
    for name, src in [
        ("qwen4_nvfp4_experts_checkpoint_test", "new_models/qwen4_exp/nvfp4_experts_test.py"),
        ("qwen4_rank_local_ft_test", "new_models/qwen4_exp/test/test_rank_local_ft.py"),
        ("qwen4_expert_factory_checkpoint_test", "new_models/qwen4_exp/expert_factory_test.py"),
        ("qwen4_mtp_input_checkpoint_test", "new_models/qwen4_exp_mtp/mtp_input_test.py"),
        ("qwen4_mtp_draft_replay_metadata_test", "new_models/qwen4_exp_mtp/draft_replay_test.py"),
        ("qwen4_mtp_vocabulary_loan_test", "new_models/qwen4_exp_mtp/vocabulary_test.py"),
    ]
]

[
    py_test(
        name = name,
        srcs = [
            "new_models/qwen4_exp/test/test_native_service.py",
            "new_models/qwen4_exp/test/test_target_weights.py",
        ] + extra,
        main = main,
        # Service publication constructs the original Qwen4 attention/cache
        # runtime and needs its accelerator compute DSO. Keep CPU tensor
        # staging, but schedule these two integration targets on CUDA workers;
        # the independent precision owner test remains on ordinary CPU workers.
        tags = ["L20_CU13"] if native_runtime else [],
        exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"} if native_runtime else {},
        env = checkpoint_reader_test_env({
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }),
        deps = [":checkpoint_reader_test_support", ":new_weight_loader", ":new_loader_core", ":models", "//rtp_llm:models", "//rtp_llm:fastsafetensors"] + (
            ["//rtp_llm/tools/convert:convert"] if name == "qwen4_mtp_native_service_checkpoint_test" else []
        ),
    )
    for name, main, extra, native_runtime in [
        ("qwen4_native_service_checkpoint_test", "new_models/qwen4_exp/test/test_native_service.py", ["new_models/qwen4_exp/test/test_precision_profiles.py"], True),
        ("qwen4_precision_source_checkpoint_test", "new_models/qwen4_exp/test/test_precision_profiles.py", ["new_models/qwen4_exp/test/test_precision_profiles.py"], False),
        ("qwen4_mtp_native_service_checkpoint_test", "new_models/qwen4_exp_mtp/native_draft_service_test.py", ["new_models/qwen4_exp_mtp/native_draft_service_test.py", "new_models/qwen4_exp/test/test_precision_profiles.py"], True),
    ]
]

models_py_test(
    name = "qwen4_static_expert_routing_test",
    src = "new_models/qwen4_exp/static_routing_test.py",
    gpu = "H20_CU13",
    gpus = 2,
    deps = [":new_weight_loader"],
)

models_py_test(
    name = "qwen4_nvfp4_tp2_marlin_hopper_test",
    src = "new_models/qwen4_exp/expert_factory_test.py",
    gpu = "H20_CU13",
    args = ["Q4SourceExpertTest.test_prequantized_tp2_marlin_slices_weights_and_scales_together"],
    env = {"RUN_Q4_NEWLOADER_MARLIN": "1"},
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
)

models_py_test(
    name = "eplb_graph_bucket_stats_hopper_test",
    src = "eplb_direct_test.py",
    gpu = "H20_CU13",
    args = ["GraphBucketStatsTest.test_changing_real_rows_in_one_captured_graph"],
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
)

models_py_test(
    name = "qwen4_nvfp4_newloader_hopper_output_test",
    src = "new_models/qwen4_exp/nvfp4_experts_test.py",
    gpu = "H20_CU13",
    args = [
        "Qwen4Nvfp4ExpertTest.test_gpu_marlin_output_against_dequantized_reference",
        "Qwen4Nvfp4ExpertTest.test_restart_maps_packed_marlin_weights_without_repacking",
    ],
    env = {"RUN_Q4_NEWLOADER_MARLIN": "1"},
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
)

models_py_test(
    name = "qwen4_mixed_storage_sm120_test",
    src = "new_models/qwen4_exp/test/test_dense_profiles.py",
    gpu = "RTX_5000_PRO_CU13",
    args = ["DenseProfileTest.test_sm120_uses_original_direct_packed_quantization"],
    env = {"RUN_Q4_DIRECT_STORAGE_GPU": "1"},
    tags = ["manual"],
    deps = [":new_weight_loader"],
)

# Shared packed layout math must not depend on either model loader.
py_library(
    name = "groupwise_export_helpers",
    srcs = [
        "read_spec.py",
        "quant_methods/groupwise_checkpoint.py",
        "quant_methods/groupwise_export.py",
    ],
    deps = ["//rtp_llm:torch"],
    visibility = ["//visibility:public"],
)

# Portable checkpoint protocol. Keep this target independent of the native
# inference library so real CPU checkpoint regressions need no fake ABI.
py_library(
    name = "new_loader_core",
    srcs = [
        "__init__.py",
        "borrowed_parameters.py",
        "downstream_heads.py",
        "expert_placement.py",
        "fastsafetensors_read_spec.py",
        "finetune_overlay.py",
        "fp4_read_spec.py",
        "fst_host_backend.py",
        "fst_range_reader.py",
        "fst_file_plan.py",
        "ft_checkpoint.py",
        "lora_checkpoint.py",
        "meta_checkpoint.py",
        "model_loader.py",
        "module_base.py",
        "quant_methods/w8a8_load_source.py",
        "read_spec.py",
        "resident_checkpoint.py",
        "modules/fp8_fp4_weight_source.py",
        "registry.py",
        "runtime_weight_update.py",
        "scratch_moe_tp.py",
        "v41_fst_file_pool.py",
        "v4_fst_nogds.py",
        "weight_mapper.py",
    ],
    deps = [
        "//rtp_llm:checkpoint_file_utils",
        "//rtp_llm/resident_weights",
        "//rtp_llm:device_detection",
        "//rtp_llm:safetensors_torch",
        "//rtp_llm:w8a8_int8_quant",
        "//rtp_llm/models_py/distributed:rank_layout",
    ] + select({
        "@//:using_ppu": [":ppu_entrypoint"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_test(
    name = "direct_lora_checkpoint_test",
    srcs = ["direct_lora_checkpoint_test.py"],
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:lora", "//rtp_llm:utils"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
)

py_test(
    name = "main_core_checkpoint_test",
    srcs = ["main_core_checkpoint_test.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    },
    deps = [":new_loader_core"],
)

py_test(
    name = "ft_module_install_test",
    srcs = ["ft_module_install_test.py"],
    deps = [":new_loader_core"],
)

py_test(
    name = "rank_local_ft_test",
    srcs = ["rank_local_ft_test.py"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm/tools/convert:ft_export_metadata",
        "//rtp_llm/tools/convert:ft_export_writer",
    ],
)

py_test(
    name = "auto_fst_route_test",
    srcs = ["auto_fst_route_test.py"],
    args = ["DefaultFstRouteTest"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    },
    deps = [":new_loader_core"],
)

# Native reader contract on build profiles that actually package the wheel.
# PPU's current wheel-less CI still runs every portable policy test above.
py_test(
    name = "auto_fst_native_reader_test",
    srcs = ["auto_fst_route_test.py"],
    main = "auto_fst_route_test.py",
    args = ["DefaultFstNativeReaderTest"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "@//:using_cuda13_arm": [],
        "@//:using_cuda12_9_x86": [],
        "@//:using_cuda12_arm": [],
        "@//:using_rocm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [":new_loader_core", "//rtp_llm:fastsafetensors"],
)

py_library(
    name = "parallel_checkpoint_test_support",
    testonly = True,
    srcs = ["parallel_checkpoint_test_support.py"],
    deps = [":new_loader_core"],
)

_GLOO_CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

py_test(
    name = "main_parallel_runtime_test",
    srcs = ["main_parallel_runtime_test.py", "layers/embedding.py"],
    env = checkpoint_reader_test_env(_GLOO_CPU_ENV),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_loader_core",
        ":parallel_checkpoint_test_support",
        "//rtp_llm:fastsafetensors",
    ],
)

# Only this integration test needs the BaseModel service descriptor.
py_test(
    name = "resident_draft_parallel_runtime_test",
    srcs = ["resident_draft_parallel_runtime_test.py"],
    data = ["//rtp_llm/libs:libth_transformer_config_so"],
    env = _GLOO_CPU_ENV,
    deps = [
        ":new_loader_core",
        ":parallel_checkpoint_test_support",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:model_descriptors",
        "//rtp_llm/resident_weights",
    ],
)

py_test(
    name = "llama_checkpoint_formats_test",
    srcs = ["llama_checkpoint_formats_test.py"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:models",
        "//rtp_llm/tools/convert:convert",
        "//rtp_llm:safetensors",
        "//rtp_llm:utils",
        "//rtp_llm/tools/convert:ft_export_metadata",
        "//rtp_llm/tools/convert:ft_export_writer",
    ],
)

py_test(
    name = "llama_meta_online_fp8_test",
    srcs = ["llama_checkpoint_formats_test.py", "llama_meta_online_fp8_test.py"],
    main = "llama_meta_online_fp8_test.py",
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    env = checkpoint_reader_test_env({}),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm:models",
        "//rtp_llm/tools/convert:ft_export_metadata",
        "//rtp_llm/tools/convert:ft_export_writer",
    ],
)

py_test(
    name = "qwen2_main_ft_stage_test",
    srcs = ["qwen2_main_ft_stage_test.py"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm/tools/convert:ft_export_metadata",
        "//rtp_llm/tools/convert:ft_export_writer",
    ],
)

py_test(
    name = "pytorch_archive_reader_test",
    srcs = ["pytorch_archive_reader_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [":new_loader_core"],
)

py_test(
    name = "resident_replay_collective_test",
    srcs = ["resident_replay_collective_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [":new_loader_core", "//rtp_llm:safetensors_torch", "//rtp_llm/resident_weights"],
)

py_test(
    name = "native_resident_restart_test",
    srcs = ["native_resident_restart_test.py"],
    deps = [":new_loader_core", "//rtp_llm:fastsafetensors", "//rtp_llm/resident_weights"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "checkpoint_source_test",
    srcs = ["checkpoint_source_test.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    },
    deps = [":new_loader_core"],
)

py_test(
    name = "v41_owned_attention_checkpoint_test",
    target_compatible_with = ue8m0_test_platforms(),
    srcs = ["new_models/deepseek_v41/attention_test.py"],
    main = "new_models/deepseek_v41/attention_test.py",
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":new_weight_loader", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_test(
    name = "v41_owned_backbone_checkpoint_test",
    target_compatible_with = ue8m0_test_platforms(),
    srcs = ["new_models/deepseek_v41/backbone_test.py"],
    main = "new_models/deepseek_v41/backbone_test.py",
    env = ue8m0_test_env(),
    deps = [
        ":new_weight_loader", "//rtp_llm:safetensors", "//rtp_llm:torch",
        "//rtp_llm/tools/convert:ft_export_metadata", "//rtp_llm/tools/convert:ft_export_writer",
    ],
)

py_test(
    name = "v41_service_checkpoint_test",
    target_compatible_with = ue8m0_test_platforms(),
    srcs = [
        "new_models/deepseek_v41/service_test.py",
        "new_models/deepseek_v41/backbone_test.py",
    ],
    main = "new_models/deepseek_v41/service_test.py",
    env = ue8m0_test_env(),
    deps = [":new_weight_loader", "//rtp_llm:models", "//rtp_llm:safetensors", "//rtp_llm:torch"],
)

# Direct parameter owners use main's existing runtime models/helpers, never a
# backported scheduler/cache/PP engine. Production routing is integrated later.
py_test(
    name = "v41_dspark_auxiliary_test",
    srcs = ["new_models/deepseek_v41_dspark/auxiliary_test.py"],
    main = "new_models/deepseek_v41_dspark/auxiliary_test.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}),
    deps = [":new_weight_loader"],
)

py_test(
    name = "v41_dspark_checkpoint_test",
    target_compatible_with = ue8m0_test_platforms(require_fst = True),
    srcs = [
        "new_models/deepseek_v41_dspark/checkpoint_test.py",
        "new_models/deepseek_v41/backbone_test.py",
    ],
    main = "new_models/deepseek_v41_dspark/checkpoint_test.py",
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":new_weight_loader"],
)

py_test(
    name = "v41_dspark_service_test",
    target_compatible_with = ue8m0_test_platforms(require_fst = True),
    srcs = [
        "new_models/deepseek_v41_dspark/service_test.py",
        "new_models/deepseek_v41_dspark/checkpoint_test.py",
        "new_models/deepseek_v41/service_test.py",
        "new_models/deepseek_v41/backbone_test.py",
    ],
    main = "new_models/deepseek_v41_dspark/service_test.py",
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "v41_dspark_cold_native_forward_test",
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    srcs = [
        "new_models/deepseek_v41_dspark/native_forward_test.py",
        "new_models/deepseek_v41_dspark/service_test.py",
        "new_models/deepseek_v41_dspark/checkpoint_test.py",
        "new_models/deepseek_v41/service_test.py",
        "new_models/deepseek_v41/backbone_test.py",
    ],
    main = "new_models/deepseek_v41_dspark/native_forward_test.py",
    data = ["//:rtp_compute_ops"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "v41_service_routes_test",
    target_compatible_with = ue8m0_test_platforms(),
    srcs = [
        "new_models/deepseek_v41/routes_test.py",
        "new_models/deepseek_v41_dspark/service_test.py",
        "new_models/deepseek_v41_dspark/checkpoint_test.py",
        "new_models/deepseek_v41/service_test.py",
        "new_models/deepseek_v41/backbone_test.py",
    ],
    main = "new_models/deepseek_v41/routes_test.py",
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "v41_dspark_release_metadata_test",
    srcs = ["new_models/deepseek_v41_dspark/release_metadata_test.py"],
    main = "new_models/deepseek_v41_dspark/release_metadata_test.py",
    tags = ["manual"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_library(
    name = "frontend_model_helpers",
    srcs = [
        "kernels/cuda/quant_layouts.py",
        "modules/__init__.py",
        "modules/dsv41/__init__.py",
        "modules/dsv41/bounded_replay.py",
        "new_models/__init__.py",
        "new_models/kimi_k25/__init__.py",
        "new_models/kimi_k25/vision_runtime.py",
        "speculative/__init__.py",
        "speculative/qwen3_dspark_checkpoint.py",
    ],
    deps = [
        ":new_loader_core",
        "//rtp_llm/models_py/new_models/qwen3_moe_eagle3:draft_body",
    ],
    visibility = ["//rtp_llm:__pkg__"],
)

py_library(
    name = "new_weight_loader",
    srcs = ["eplb_controller.py", "new_weight_manager.py", "quantized_weight_update.py"] + glob([
        "layers/*.py",
        "new_models/*.py",
        "new_models/**/*.py",
        "quant_methods/*.py",
    ], exclude = [
        "quant_methods/w8a8_load_source.py",
        "**/test/**",
        "**/tests/**",
        "**/*_test.py",
        "**/test_*.py",
    ]),
    # Full model owners import both native binding modules. Place the DSOs
    # together in the package path used by rtp_llm.ops; the portable reader
    # above deliberately has no native inference dependency.
    data = [
        "//rtp_llm/libs:librtp_compute_ops_so",
        "//rtp_llm/libs:libth_transformer_config_so",
    ],
    # These are top-level Python extensions, not rtp_llm.libs submodules.
    # A direct kernel import must work before rtp_llm.ops initializes sys.path.
    imports = ["../libs"],
    deps = [
        ":models",
        ":new_loader_core",
        "//rtp_llm/models_py/new_models/qwen3_moe_eagle3:draft_body",
        "//rtp_llm:config",
        "//rtp_llm:eplb",
        "//rtp_llm/model_loader/tipc:tipc",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:ops",
        "//rtp_llm:triton",
        "//rtp_llm/models_py/distributed:collective_torch",
    ] + select({
        # The ROCm fastsafetensors wrapper imports fast_safetensors, shipped
        # in a second wheel that its metadata does not declare as a dependency.
        "@//:using_rocm": ["//rtp_llm:fast-safetensors"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "int8_export_helpers",
    srcs = ["quant_methods/int8.py"],
    deps = [":groupwise_export_helpers", "//rtp_llm:torch"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "main_update_compat_test",
    srcs = ["main_update_compat_test.py"],
    deps = [":new_weight_loader"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""},
)

py_test(
    name = "serving_route_checkpoint_test",
    srcs = ["serving_route_checkpoint_test.py"],
    env = checkpoint_reader_test_env({}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models"],
)

[
    py_test(
        name = name,
        srcs = [entry],
        main = entry,
        deps = [":new_weight_loader"],
        env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""},
    )
    for name, entry in [
        ("v4_checkpoint_namespace_test", "new_models/deepseek_v4/checkpoint_namespace_test.py"),
        ("v4_load_geometry_test", "new_models/deepseek_v4/load_geometry_test.py"),
    ]
]

[
    py_test(
        name = name,
        srcs = [entry] + extra,
        main = entry,
        deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models"] + (["//rtp_llm/tools/convert:ft_export_engram"] if name == "v41_engram_source_test" else []),
        env = {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "NEWLOADER_TEST_NATIVE_FST": "1",
            "OMP_NUM_THREADS": "1",
        },
        target_compatible_with = ue8m0_test_platforms(require_fst = True),
    )
    for name, entry, extra in [
        ("v41_engram_source_test", "new_models/deepseek_v41/engram_source_test.py", []),
        ("v41_linear_owner_test", "new_models/deepseek_v41/linear_owner_test.py", []),
        ("deepseek_v4_hc_load_test", "new_models/deepseek_v4/hc_test.py", []),
        ("deepseek_v4_backbone_load_test", "new_models/deepseek_v4/backbone_test.py", []),
        ("deepseek_v4_mtp_load_test", "new_models/deepseek_v4/mtp_test.py", [
            "new_models/deepseek_v4/backbone_test.py",
            "serving_route_checkpoint_test.py",
        ]),
        ("v4_dspark_checkpoint_routes_test", "new_models/deepseek_v4_dspark/checkpoint_test.py", []),
        ("v4_dspark_aux_checkpoint_test", "new_models/deepseek_v4_dspark/aux_checkpoint_test.py", []),
        ("v4_dspark_owner_checkpoint_test", "new_models/deepseek_v4_dspark/owner_checkpoint_test.py", [
            "new_models/deepseek_v4/backbone_test.py",
            "serving_route_checkpoint_test.py",
        ]),
    ]
]

py_test(
    name = "deepseek_v4_mtp_cuda13_service_test",
    srcs = [
        "new_models/deepseek_v4/mtp_test.py",
        "new_models/deepseek_v4/backbone_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "new_models/deepseek_v4/mtp_test.py",
    args = ["V4MtpCheckpointTest.test_real_gpu_service_publication"],
    env = {"V4_MTP_SERVICE_GPU": "cuda:0"},
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "bert_input_vocab_test",
    srcs = ["bert_input_vocab_test.py", "bert_service_test.py", "serving_route_checkpoint_test.py"],
    main = "bert_input_vocab_test.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models", "//rtp_llm:transformers"],
)

checkpoint_service_tests(
    name = "bert_service_test",
    native_name = "bert_native_output_test",
    cpu_case = "BertServiceTest",
    native_case = "BertNativeOutputTest",
    srcs = [
        "bert_service_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "bert_service_test.py",
    env = checkpoint_reader_test_env({
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm:models",
        "//rtp_llm:transformers",
    ],
)

py_test(
    name = "jina_direct_checkpoint_cpu_test",
    srcs = ["new_models/jina_bert/jina_checkpoint_test.py"],
    main = "new_models/jina_bert/jina_checkpoint_test.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
)

checkpoint_service_tests(
    name = "jina_native_checkpoint_test",
    native_name = "jina_native_output_test",
    cpu_case = "JinaServiceCheckpointTest",
    native_case = "JinaRuntimeCheckpointTest",
    srcs = [
        "new_models/jina_bert/jina_checkpoint_test.py",
        "new_models/jina_bert/jina_runtime_checkpoint_test.py",
    ],
    main = "new_models/jina_bert/jina_runtime_checkpoint_test.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "qwen3_dspark_newloader_checkpoint_test",
    srcs = ["new_models/qwen3_dspark/checkpoint_test.py"],
    main = "new_models/qwen3_dspark/checkpoint_test.py",
    env = checkpoint_reader_test_env({}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
)

py_test(
    name = "qwen3_dspark_checkpoint_cuda_graph_test",
    srcs = [
        "new_models/qwen3_dspark/checkpoint_cuda_test.py",
        "new_models/qwen3_dspark/checkpoint_test.py",
    ],
    main = "new_models/qwen3_dspark/checkpoint_cuda_test.py",
    env = checkpoint_reader_test_env({}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "qwen3_dspark_checkpoint_format_test",
    srcs = [
        "new_models/qwen3_dspark_format_test.py",
        "speculative/qwen3_dspark_checkpoint.py",
    ],
    main = "new_models/qwen3_dspark_format_test.py",
    deps = ["//rtp_llm:safetensors", "//rtp_llm:torch"],
)

py_test(
    name = "encoder_qwen3_ft_test",
    env = checkpoint_reader_test_env(),
    srcs = [
        "new_models/encoder_qwen3_ft_test.py",
        "new_models/jina_bert/jina_checkpoint_test.py",
        "new_models/qwen3_dspark/checkpoint_test.py",
    ],
    main = "new_models/encoder_qwen3_ft_test.py",
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models", "//rtp_llm/tools/convert:convert"],
)

py_test(
    name = "multimodal_dense_ft_test",
    srcs = ["new_models/multimodal_dense_ft_test.py"],
    main = "new_models/multimodal_dense_ft_test.py",
    env = checkpoint_reader_test_env(),
    deps = [":new_weight_loader", "//rtp_llm:models", "//rtp_llm/tools/convert:convert"],
)

py_test(
    name = "qwen3_dspark_sampler_checkpoint_test",
    srcs = ["new_models/qwen3_dspark_weights_test.py"],
    main = "new_models/qwen3_dspark_weights_test.py",
    deps = [":new_weight_loader"],
)

checkpoint_service_tests(
    name = "qwen3_dspark_native_service_test",
    env = checkpoint_reader_test_env({}),
    native_name = "qwen3_dspark_native_output_test",
    cpu_case = "PublicDraftLoadingTest",
    native_case = "DraftNativeOutputTest",
    srcs = [
        "new_models/qwen3_dspark/checkpoint_test.py",
        "qwen3_dspark_native_service_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "qwen3_dspark_native_service_test.py",
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:async_model", "//rtp_llm:model_factory", "//rtp_llm:models", "//rtp_llm:transformers"],
)

py_test(
    name = "serving_route_qwen3_test",
    srcs = [
        "serving_route_qwen3_test.py",
        "serving_route_checkpoint_test.py",
        "new_models/qwen3/main_pp_checkpoint_test.py",
    ],
    main = "serving_route_qwen3_test.py",
    env = checkpoint_reader_test_env({}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "main_native_pp_checkpoint_test",
    srcs = ["new_models/qwen3/main_pp_checkpoint_test.py"],
    main = "new_models/qwen3/main_pp_checkpoint_test.py",
    env = checkpoint_reader_test_env({
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    }),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm:transformers",
    ],
)

py_test(
    name = "glm5_packed_mega_eplb_sm100_test",
    srcs = ["new_models/glm5/mega_eplb_sm100_test.py"],
    main = "new_models/glm5/mega_eplb_sm100_test.py",
    deps = [":new_weight_loader", "//rtp_llm:models"],
    tags = ["SM100_ARM_CU13"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "glm5_packed_mega_eplb_test",
    srcs = ["new_models/glm5/mega_eplb_test.py"],
    main = "new_models/glm5/mega_eplb_test.py",
    env = {"CUDA_VISIBLE_DEVICES": ""},
    deps = [":new_weight_loader", "//rtp_llm:models"],
    target_compatible_with = ue8m0_test_platforms(),
)

py_test(
    name = "eplb_direct_reload_test",
    srcs = ["eplb_direct_test.py"],
    main = "eplb_direct_test.py",
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
    timeout = "moderate",
)

py_test(
    name = "staged_moe_collective_order_test",
    srcs = ["staged_moe_collective_order_test.py"],
    main = "staged_moe_collective_order_test.py",
    deps = [":new_weight_loader"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    timeout = "moderate",
)

py_test(
    name = "qwen3_moe_checkpoint_test",
    srcs = [
        "qwen3_moe_checkpoint_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "qwen3_moe_checkpoint_test.py",
    env = checkpoint_reader_test_env({}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "qwen3_moe_native_service_test",
    srcs = [
        "qwen3_moe_native_service_test.py",
        "qwen3_moe_checkpoint_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "qwen3_moe_native_service_test.py",
    env = checkpoint_reader_test_env({}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models"],
)

checkpoint_service_tests(
    name = "qwen3_vl_native_service_test",
    native_name = "qwen3_vl_native_output_test",
    cpu_case = "PublicVLTest",
    native_case = "VLNativeOutputTest",
    srcs = [
        "new_models/qwen3_vl/service_test.py",
        "qwen3_moe_checkpoint_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "new_models/qwen3_vl/service_test.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm:models",
        "//rtp_llm:multimodal",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
    ],
)

py_test(
    name = "qwen3_vl_device_rotary_test",
    srcs = [
        "qwen3_vl_device_rotary_test.py",
        "new_models/qwen3_vl/service_test.py",
        "qwen3_moe_checkpoint_test.py",
        "serving_route_checkpoint_test.py",
    ],
    tags = ["H20_CU13"],
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:models",
        "//rtp_llm:multimodal",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
    ],
)

py_test(
    name = "serving_route_deepseek_test",
    srcs = [
        "serving_route_checkpoint_test.py",
        "serving_route_deepseek_test.py",
    ],
    main = "serving_route_deepseek_test.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models"],
)

py_library(
    name = "owner_migration_checkpoint_test_cases",
    testonly = True,
    srcs = ["owner_migration_checkpoint_test.py"],
    deps = [":new_weight_loader"],
    visibility = ["//rtp_llm/model_loader/test:__pkg__"],
)

py_library(
    name = "ple_residency_test_cases",
    testonly = True,
    srcs = ["new_models/qwen4_exp/ple_residency_test.py"],
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
    visibility = ["//rtp_llm/models/qwen4_exp/test:__pkg__"],
)

py_test(
    name = "ple_host_fst_routing_test",
    srcs = ["new_models/qwen4_exp/ple_residency_test.py"],
    main = "new_models/qwen4_exp/ple_residency_test.py",
    env = checkpoint_reader_test_env({}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader"],
)

py_test(
    name = "v4_wo_a_layout_test",
    srcs = [
        "modules/dsv4/fp8/_wo_a_layout.py",
        "v4_wo_a_layout_test.py",
    ],
    main = "v4_wo_a_layout_test.py",
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "owner_migration_checkpoint_test",
    srcs = ["owner_migration_checkpoint_test.py"],
    deps = [":new_weight_loader"],
)

exports_files(
    ["modules/factory/linear/fixed_m_linear.py"],
    visibility = ["//rtp_llm/models_py/modules/factory/linear/test:__pkg__"],
)

py_library(
    name = "utils",
    srcs = glob([
        "utils/*.py",
    ]),
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "qwen_service_vision_test",
    srcs = [
        "new_models/qwen_service_vision.py",
        "new_models/qwen_service_vision_test.py",
        "new_models/qwen_visual_namespace.py",
    ],
    main = "new_models/qwen_service_vision_test.py",
    env = checkpoint_reader_test_env({
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    }),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_loader_core",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:transformers",
    ],
)

py_test(
    name = "megatron_bert_checkpoint_test",
    srcs = ["new_models/megatron_bert/test_checkpoint.py"],
    main = "new_models/megatron_bert/test_checkpoint.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:fastsafetensors", "//rtp_llm:transformers"],
)

checkpoint_service_tests(
    name = "megatron_bert_native_checkpoint_test",
    native_name = "megatron_bert_native_output_test",
    cpu_case = "MegatronServiceCheckpointTest",
    native_case = "NativeMegatronTest",
    srcs = [
        "new_models/megatron_bert/native_checkpoint_test.py",
        "new_models/megatron_bert/test_checkpoint.py",
    ],
    main = "new_models/megatron_bert/native_checkpoint_test.py",
    env = checkpoint_reader_test_env({"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:fastsafetensors", "//rtp_llm:models", "//rtp_llm:transformers"],
)

# CPU-only subset of the DSv4 decode package: pure-torch modules plus the
# package __init__ chain, without the CUDA/testlib closure. The lazily
# imported paged/pool helpers (pool_slot_mapping, _pool_triton) are part of
# the closure: without them, enabling the paged or is_cuda branches under
# sandboxed runfiles raises ModuleNotFoundError instead of running the
# (pure-torch / triton-optional) helper.
py_library(
    name = "dsv4_decode_cpu",
    srcs = [
        "modules/__init__.py",
        "modules/dsv4/__init__.py",
        "modules/dsv4/_pool_triton.py",
        "modules/dsv4/decode/__init__.py",
        "modules/dsv4/decode/decode_attn_metadata.py",
        "modules/dsv4/decode/kv_write_decode_op.py",
        "modules/dsv4/fp8/__init__.py",
        "modules/dsv4/fp8/decode/__init__.py",
        "modules/dsv4/fp8/decode/pool_slot_mapping.py",
        "modules/dsv4/kv_cache_utils.py",
    ],
    data = [
        "//rtp_llm:__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm:torch",
    ],
)

# flashinfer-python ships in the cuda12 and cuda13 locks
flashinfer = ["flashinfer-python"]
deep = ["deep_gemm", "deep_ep"]
flashmla = ["fast-hadamard-transform", "flash-mla", "tilelang"]
rtp_kernel = ["rtp-kernel"]
flashattn2 = ["flash_attn"]
flashattn = flashattn2 + ["flash-attn-3"]
requirement(flashinfer)
requirement(deep)
requirement(flashmla)
requirement(["flash-mla-legacy"])
requirement(rtp_kernel)
requirement(flashattn)

filegroup(
    name = "cutlass_moe_config",
    srcs = glob(["kernels/cuda/fp8_kernel/cutlass_groupgemm/*"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "kernel_layout_sources",
    srcs = [
        "kernels/__init__.py",
        "kernels/cuda/fp8_input_quant.py",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rocm_fp8_ptpc_hipb_solutions",
    srcs = [
        "modules/factory/linear/impl/rocm/data/fp8_ptpc_hipb_solutions.json",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "tile_kernels_mhc",
    srcs = glob([
        "3rdparty/__init__.py",
        "3rdparty/tile_kernels/*.py",
        "3rdparty/tile_kernels/**/*.py",
    ]),
    # The mhc kernels import tilelang at module scope; carry the wheel (and
    # its z3-solver/apache-tvm-ffi closure, via the pip pkg deps) in runfiles
    # instead of relying on the image's site-packages. Only the locks used by
    # the DSv4 lanes ship tilelang.
    deps = select({
        "@//:using_cuda13_x86": ["//rtp_llm:tilelang"],
        "@//:using_cuda13_arm": ["//rtp_llm:tilelang"],
        "@//:using_cuda12_9_x86": ["//rtp_llm:tilelang"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "dsv41_jit_sources",
    srcs = glob([
        "modules/dsv41/**/*.cu",
        "modules/dsv41/**/*.cuh",
        "modules/dsv41/**/*.cc",
        "modules/dsv41/**/*.h",
        "modules/dsv41/**/*.LICENSE",
    ]),
    visibility = ["//visibility:public"],
)

_DSV4_PPU_MODULES = [
    "modules/dsv4/ppu/**/*.py",
    "modules/factory/linear/ppu_impl/fp8_linear.py",
]
_DSV4_PPU_KERNELS = [
    "kernels/ppu/attention/**/*.py", "kernels/ppu/moe/**/*.py",
    "kernels/ppu/hc/**/*.py", "kernels/ppu/mtp/**/*.py",
    "kernels/ppu/jit/**/*.py", "kernels/ppu/runtime.py",
]

cc_library(
    name = "marlin_nvfp4_kernels",
    srcs = [
        "kernels/cuda/marlin_nvfp4/marlin_kernels/schema.cpp",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/libtorch_stable/moe/marlin_moe_wna16/ops.cu",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu",
    ],
    hdrs = glob([
        "kernels/cuda/marlin_nvfp4/marlin_kernels/**/*.h",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/**/*.hpp",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/**/*.cuh",
    ]),
    includes = ["kernels/cuda/marlin_nvfp4/marlin_kernels"],
    # Keep the JIT's compilation semantics and emit both supported cubins,
    # independently of the engine's selected GPU architecture.
    copts = cuda_default_copts_without_arch() + [
        "-O3",
        "-std=c++17",
        "-DUSE_CUDA",
        # Match the runtime admission's minimum stable Torch ABI (2.10).
        "-DTORCH_TARGET_VERSION=0x020a000000000000",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-nvcc_options=expt-relaxed-constexpr",
        "-nvcc_options=static-global-template-stub=false",
        "-nvcc_options=ftz=false",
        "--cuda-gpu-arch=sm_90",
        "--cuda-gpu-arch=sm_120",
    ],
    alwayslink = True,
    target_compatible_with = select({
        "//:using_cuda13_x86": [],
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        "//:using_cuda13_x86": ["@torch_2.11_py310_cuda//:torch"],
        "//:using_cuda13_arm": ["@torch_2.11_py310_cuda-aarch64//:torch"],
        "//conditions:default": [],
    }) + [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "marlin_nvfp4",
    linkopts = [
        "-Wl,--as-needed",
        "-Wl,-rpath='$$ORIGIN/../../torch/lib'",
    ],
    linkshared = True,
    deps = [":marlin_nvfp4_kernels"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "modules",
    srcs = glob([
        "modules/*.py",
        "modules/**/*.py",
    ], exclude = _DSV4_PPU_MODULES + ["modules/qwen4_exp/test/**"]) + select({
        "@//:using_ppu": glob(_DSV4_PPU_MODULES),
        "//conditions:default": [],
    }),
    data = [
        ":cutlass_moe_config",
        ":dsv41_jit_sources",
        ":rocm_fp8_ptpc_hipb_solutions",
    ] + select({
        "@//:using_ppu": [":ppu_sources"],
        "//conditions:default": [],
    }),
    deps = [
        ":new_loader_core",
        ":utils",
        ":kernels",
        ":tile_kernels_mhc",
        "//rtp_llm:warmup",
        "//rtp_llm/models_py/kernel_tuning",
        "//rtp_llm/resident_weights",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm/models_py/distributed:deepep_wrapper",
        "//rtp_llm/models_py/distributed:moriep_wrapper",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/models_py/distributed:cpu_status",
        "//rtp_llm/models_py/distributed:rank_layout",
        "//rtp_llm/models_py/distributed:user_buffers",
    ] + select({
        # cuda13 locks ship flashinfer-python and deep-ep.
        "@//:using_cuda13_x86": flashinfer + deep,
        "@//:using_cuda13_arm": flashinfer + deep,
        "@//:using_cuda12": flashinfer + deep,
        "@//:using_ppu": deep,
        "@//:using_rocm": [
            "//rtp_llm/models_py/modules/base/rocm:rocm_base",
            "//rtp_llm/models_py/modules/base/rocm:trt_allreduce",
        ],
        "//conditions:default": [],
    }) + select({
        # cuda12_arm (kept from main) has no branch here: its lock ships none of these.
        "@//:using_cuda13_x86": ["flash-mla", "fast-hadamard-transform", "tilelang"],
        "@//:using_cuda13_arm": ["flash-mla", "flash-mla-legacy", "tilelang"],
        "@//:using_cuda12_9_x86": flashmla,
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda13_x86": rtp_kernel,
        "@//:using_cuda13_arm": rtp_kernel,
        "@//:using_cuda12_9_x86": rtp_kernel,
        "@//:using_cuda12_arm": rtp_kernel,
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda13_x86": flashattn2,
        "@//:using_cuda13_arm": [],
        "@//:using_cuda12_9_x86": flashattn,
        "@//:using_cuda12_arm": flashattn,
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "dsv4_fused_kernel_sources",
    srcs = [
        "modules/dsv4/_fused_inv_rope_fp8_quant_triton.py",
        "modules/dsv4/_fused_rmsnorm_rope_triton.py",
        "modules/dsv4/_rope_only_triton.py",
    ],
    visibility = ["//visibility:public"],
)

# Export sitecustomize.py for test targets to fix nvidia-cutlass-dsl import
exports_files(["modules/factory/sitecustomize.py"])

exports_files(
    [
        "new_models/deepseek_v4/ppu_platform.py",
        "modules/dsv4/ppu/resources.py",
        "modules/dsv4/ppu/moe/ppu_ep_moe.py",
        "modules/dsv4/ppu/metadata/cache.py",
        "modules/dsv4/ppu/manifest.py",
    ],
    visibility = ["//rtp_llm/models_py/modules/dsv4/ppu/test:__pkg__"],
)

exports_files(
    ["new_models/deepseek_v4/backbone_test.py"],
    visibility = ["//rtp_llm/models_py/new_models/deepseek_v4/test:__pkg__"],
)

py_library(
    name = "modules_cuda",
    srcs = glob([
        "modules/cuda/*.py",
        "modules/cuda/**/*.py",
    ]),
    visibility = [":__subpackages__"],
)

py_library(
    name = "modules_rocm",
    srcs = glob([
        "modules/rocm/*.py",
        "modules/rocm/**/*.py",
    ]),
    visibility = [":__subpackages__"],
)

py_library(
    name = "ppu_entrypoint",
    srcs = [
        "__init__.py",
        "modules/factory/attention/register.py",
        "modules/factory/fused_moe/register.py",
        "modules/factory/linear/register.py",
    ],
    deps = [
        "//rtp_llm:device",
        "//rtp_llm:utils",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "ppu_modules",
    # Keep the public PPU test closure aligned with the internal ppu_modules
    # target: the factory imports the shared model runtime during collection.
    srcs = [],
    deps = [
        ":ppu_entrypoint",
        ":modules",
        "//rtp_llm:models",
        "//rtp_llm:config",
        "//rtp_llm:device",
        "//rtp_llm:ops",
        "//rtp_llm:lora",
        "//rtp_llm:transformers",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm:pydantic",
        "//rtp_llm:utils",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//rtp_llm/libs:libth_transformer_config_so",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "scaled_rope",
    srcs = ["kernels/cuda/scaled_rope.py"],
    deps = ["//rtp_llm:torch"] + triton_deps(["triton"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "kernels",
    srcs = glob([
        "kernels/*.py",
        "kernels/**/*.py",
    ], exclude = _DSV4_PPU_KERNELS + ["kernels/**/test/**", "kernels/cuda/scaled_rope.py"]) + select({
        "@//:using_ppu": glob(_DSV4_PPU_KERNELS),
        "//conditions:default": [],
    }),
    data = [
        "//rtp_llm/libs:marlin_nvfp4",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/LICENSE",
    ] + glob(["kernels/cuda/qsa/*.cu", "kernels/cuda/qsa/*.cuh"]),
    deps = [
        ":scaled_rope",
        ":utils",
        "//rtp_llm/resident_weights",
        "//rtp_llm:torch",
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + triton_deps(["triton"]),
    visibility = ["//rtp_llm/models_py:__subpackages__"],
)

py_library(
    name = "internal_kernels",
    srcs = glob(["kernels/ppu/**/*.py"], exclude = _DSV4_PPU_KERNELS) + select({
        "@//:using_ppu": glob(_DSV4_PPU_KERNELS),
        "//conditions:default": [],
    }),
    data = select({
        "@//:using_ppu": [":ppu_jit_sources"],
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "configs",
    srcs = glob([
        "configs/*.py",
    ]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "models",
    srcs = glob([
        "model_desc/*.py",
        "model_desc/cuda_graph/*.py",
        "model_desc/qwen4_exp_runtime/*.py",
        "speculative/*.py",
    ]) + select({
        "@//:using_ppu": glob(["model_desc/deepseek_v4_ppu/**/*.py"]),
        "//conditions:default": [],
    }),
    deps = [
        ":utils",
        ":modules",
        ":kernels",
        ":configs",
        "//rtp_llm:config",
        "//rtp_llm:device",
        "//rtp_llm:warmup",
        "//rtp_llm/model_loader:loader",
    ] + select({
        "@//:using_cuda12": [
            ":modules_cuda",
        ],
        "@//:using_rocm": [
            ":modules_rocm"
        ],
        "//conditions:default": [],
    }) + select({
        # deepseek_v4_model.py lazily imports modules/dsv4/tilelang_kernels and
        # the DSv4 warmup reaches the mhc TileLang kernels; declare tilelang on
        # the owning library so its runfiles do not depend on the image's
        # site-packages install. z3-solver/apache-tvm-ffi ride along through the
        # tilelang pip pkg deps.
        "@//:using_cuda13_x86": ["//rtp_llm:tilelang"],
        "@//:using_cuda13_arm": ["//rtp_llm:tilelang"],
        "@//:using_cuda12_9_x86": ["//rtp_llm:tilelang"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# PPU-native JIT sources retain their frozen relative include layout.
filegroup(
    name = "ppu_jit_sources",
    srcs = glob(["kernels/ppu/**/*.cuh", "kernels/ppu/**/*.h", "kernels/ppu/**/*.json", "kernels/ppu/**/*.patch", "kernels/ppu/**/NOTICE*"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "ppu_sources",
    srcs = glob([
        "model_desc/deepseek_v4_ppu/**/*.py",
        "modules/dsv4/ppu/**/*.py",
        "modules/factory/linear/ppu_impl/*.py",
        "modules/factory/fused_moe/ppu_impl/*.py",
        "kernels/ppu/**/*.py",
    ], exclude = ["**/test/**", "**/tests/**"]) + [":ppu_jit_sources", "//rtp_llm:models/deepseek_v4.py"],
    visibility = ["//visibility:public"],
)

exports_files(
    ["model_desc/cp_mtp_hidden_handoff.py"],
    visibility = ["//rtp_llm/models_py/model_desc/test:__pkg__"],
)

checkpoint_service_tests(
    name = "roberta_checkpoint_test",
    native_name = "roberta_native_output_test",
    cpu_case = "RobertaCheckpointTest",
    native_case = "RobertaNativeOutputTest",
    srcs = ["new_models/roberta/test_roberta_checkpoint.py"],
    main = "new_models/roberta/test_roberta_checkpoint.py",
    env = checkpoint_reader_test_env({
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }),
    deps = [
        ":checkpoint_reader_test_support",
        ":new_weight_loader",
        "//rtp_llm:models",
        "//rtp_llm:transformers",
    ],
)

py_test(
    name = "bert_input_vocab_native_output_test",
    srcs = [
        "bert_input_vocab_native_output_test.py",
        "bert_input_vocab_test.py",
        "bert_service_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "bert_input_vocab_native_output_test.py",
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    env = checkpoint_reader_test_env({
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }),
    deps = [":checkpoint_reader_test_support", ":new_weight_loader", "//rtp_llm:models", "//rtp_llm:transformers"],
)
