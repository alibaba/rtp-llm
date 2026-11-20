load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")
load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")
load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_sm100_py_test")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_v41_native_cache_specs",
    srcs = ["test_v41_native_cache_specs.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
    data = ["//rtp_llm/cpp/cache/test:cache_config_creator_py_test"],
    tags = ["dsv41_cuda13"],
    env = {"GPU_COUNT": "0"},
    legacy_create_init = 0,
)

[
    py_test(
        name = name + "_cpu",
        srcs = [name + ".py"],
        main = name + ".py",
        deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
        tags = ["dsv41_cuda13"],
        env = {"GPU_COUNT": "0"},
        legacy_create_init = 0,
    )
    for name in ["test_cache_specs", "test_cache_tags"]
]

[
    cuda13_sm100_py_test(
        name = name,
        srcs = ["test_v41_cp_collectives.py"],
        main = "test_v41_cp_collectives.py",
        gpu_count = gpu_count,
        deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
        env = {"DSV41_CP_TEST_WORLD_SIZE": str(gpu_count)},
        tags = ["dsv41_cuda13"],
        legacy_create_init = 0,
    )
    for name, gpu_count in [
        ("test_v41_cp_collectives_cp2", 2),
        ("test_v41_cp_collectives", 4),
        ("test_v41_cp_collectives_cp8", 8),
    ]
]

cuda13_sm100_py_test(
    name = "test_engram_scale_shards",
    srcs = glob(["*.py"]) + ["//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "3", "$(location :test_engram_scale_shards.py)"],
    gpu_count = 3,
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

py_test(
    name = "test_prefill_positions_upload",
    srcs = ["test_prefill_positions_upload.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
    tags = ["dsv41_cuda13", "SM103_TEST"],
    exec_properties = {"gpu": "SM103_TEST", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "test_v41_front_gemm",
    srcs = ["test_v41_front_gemm.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
    tags = ["dsv41_cuda13", "SM103_TEST"],
    exec_properties = {"gpu": "SM103_TEST", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

_ENGRAM_SM100_TESTS = [
    "test_engram",
    "test_engram_cpu",
    "test_engram_cpu_deadline",
    "test_engram_cpu_native_packed",
    "test_engram_cuda",
    "test_engram_packed",
    "test_engram_prefetch",
    "test_engram_decode_schedule",
    "test_engram_quantized_slice",
]

_TESTS = [
    "test_cp_metadata_high_batch",
    "test_dspark_commit_wkv",
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
    "test_mhc_moe_inputs",
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
    "test_v41_aux_capture_mhc",
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
    for name in _TESTS if name not in ["test_prefill_workspace", "test_strategy_select", "test_gate_routing_metadata", "test_v41_aux_capture_mhc", "test_v41_cross_layer_mhc", "test_v41_mega_mhc", "test_mega_gate", "test_mhc_moe_inputs", "test_v41_query_quant", "test_silu_mul_bf16", "test_dsv41_kernel_jit_warmup", "test_v41_decode_rope_metadata", "test_v41_mhc_prenorm", "test_v41_hc_jit_warmup", "test_mhc_lazy_backward", "test_v41_mega_mhc_compact"] + _ENGRAM_SM100_TESTS
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
    for name in ["test_v41_hc_jit_warmup", "test_mhc_lazy_backward", "test_v41_mega_mhc_compact", "test_v41_mhc_prenorm", "test_v41_decode_rope_metadata", "test_silu_mul_bf16", "test_prefill_workspace", "test_strategy_select", "test_gate_routing_metadata", "test_v41_aux_capture_mhc", "test_v41_cross_layer_mhc", "test_v41_mega_mhc", "test_mega_gate", "test_private_gate_pack_eplb", "test_mhc_moe_inputs", "test_fake_balance_expert", "test_v41_query_quant", "test_dsv41_kernel_jit_warmup", "test_engram_decode_prefetch", "test_engram_quantized"] + _ENGRAM_SM100_TESTS
]

dsv4_sm100_py_test(
    name = "test_eplb_contract",
    gpu_count = 0,
    tags = ["dsv41_cuda13"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

dsv4_sm100_py_test(
    name = "test_eplb_mega",
    gpu_count = 2,
    tags = ["dsv41_cuda13"],
    extra_srcs = ["//rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test:mega_moe_routed_only_execution_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
    ],
)

py_test(
    name = "test_sm90_gate_bf16_forward",
    srcs = ["test_sm90_gate_bf16_forward.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_gate_bf16_forward.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_gate_bf16_input",
    srcs = ["test_sm90_gate_bf16_input.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_gate_bf16_input.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_mega_moe",
    srcs = ["test_sm90_mega_moe.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "0", "$(location :test_sm90_mega_moe.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", test_requirement("pytest")],
    env = {"GPU_COUNT": "0"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_dispatch_pad",
    srcs = ["test_sm90_dispatch_pad.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_dispatch_pad.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_shared_overlap",
    srcs = ["test_sm90_shared_overlap.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_shared_overlap.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_quant_scale_fused",
    srcs = ["test_sm90_quant_scale_fused.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_quant_scale_fused.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_engram_registration_ranges",
    srcs = ["test_sm90_engram_registration_ranges.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_engram_registration_ranges.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_engram_uva_prefetch",
    srcs = ["test_sm90_engram_uva_prefetch.py", "test_sm90_engram.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_engram_uva_prefetch.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_deepgemm_dense",
    srcs = ["test_sm90_deepgemm_dense.py", "sm90_dense_reference.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_deepgemm_dense.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_deepgemm_reference_stress",
    srcs = ["test_sm90_deepgemm_reference_stress.py", "test_sm90_deepgemm_dense.py", "sm90_dense_reference.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_deepgemm_reference_stress.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_reference_stress", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_decode_pool_specs",
    srcs = ["test_decode_pool_specs.py"],
    main = "test_decode_pool_specs.py",
    deps = ["//rtp_llm/models_py:modules"],
    env = {"GPU_COUNT": "0"},
    tags = ["dsv41_cuda13", "H20_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_mxfp8_dense",
    srcs = ["test_sm90_mxfp8_dense.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_mxfp8_dense.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_dsv41_kernel_jit_warmup_sm90",
    srcs = ["test_dsv41_kernel_jit_warmup.py"],
    main = "test_dsv41_kernel_jit_warmup.py",
    deps = ["//rtp_llm/models_py:modules"],
    env = {"GPU_COUNT": "0"},
    tags = ["dsv41_cuda13", "H20_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_delayed_hc",
    srcs = ["test_sm90_delayed_hc.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_delayed_hc.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:tile_kernels_mhc",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_mega_sm90_input_pack",
    srcs = ["test_mega_sm90_input_pack.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "0", "$(location :test_mega_sm90_input_pack.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", test_requirement("pytest")],
    env = {"GPU_COUNT": "0"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_mega_sm90_input_pack_cuda",
    srcs = ["test_mega_sm90_input_pack_cuda.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_mega_sm90_input_pack_cuda.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_mhc_sm90_mega",
    srcs = ["test_mhc_sm90_mega.py"],
    main = "test_mhc_sm90_mega.py",
    deps = ["//rtp_llm/models_py:modules"],
    env = {"GPU_COUNT": "0"},
    tags = ["dsv41_cuda13", "H20_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_mhc_sm90_mega_cuda",
    srcs = ["test_mhc_sm90_mega_cuda.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_mhc_sm90_mega_cuda.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py:tile_kernels_mhc",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_engram",
    srcs = ["test_sm90_engram.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_engram.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

[
    py_test(
        name = name + "_sm90",
        srcs = glob(["*.py"]) + ["//rtp_llm/test/pytest:pytest_main.py"],
        main = "//rtp_llm/test/pytest:pytest_main.py",
        args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :" + name + ".py)"],
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:modules",
            "//rtp_llm/models_py:tile_kernels_mhc",
            "//rtp_llm:jit_cache_env",
            test_requirement("pytest"),
        ],
        tags = ["dsv41_cuda13", "H20_CU13"],
        env = {"GPU_COUNT": "1"},
        exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
        target_compatible_with = select({
            "@//:using_cuda13_x86": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        legacy_create_init = 0,
    )
    for name in [
        "test_v41_mhc_prenorm",
        "test_v41_hc_jit_warmup",
        "test_v41_delayed_hc_cuda_graph",
    ]
]

py_test(
    name = "test_h20_cast_elision",
    srcs = ["test_h20_cast_elision.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_h20_cast_elision.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_quant_scale_mn_major",
    srcs = ["test_sm90_quant_scale_mn_major.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_quant_scale_mn_major.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)
