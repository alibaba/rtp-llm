load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_sm90_flashinfer_glue_integration",
    srcs = ["test_sm90_flashinfer_glue_integration.py", "test_sm90_flashinfer_checkpoint.py", "test_sm90_flashinfer_local.py", "sm90_flashinfer_reference.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_flashinfer_glue_integration.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", "//rtp_llm:safetensors", "@flashinfer_opensource_sidecar//:pkg", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_flashinfer_glue",
    srcs = ["test_sm90_flashinfer_glue.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_flashinfer_glue.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_flashinfer_meta_init",
    srcs = ["test_sm90_flashinfer_meta_init.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_flashinfer_meta_init.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", "@flashinfer_opensource_sidecar//:pkg", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_flashinfer_checkpoint",
    srcs = ["test_sm90_flashinfer_checkpoint.py", "test_sm90_flashinfer_local.py", "sm90_flashinfer_reference.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_flashinfer_checkpoint.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", "//rtp_llm:safetensors", "@flashinfer_opensource_sidecar//:pkg", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_flashinfer_perf",
    srcs = ["test_sm90_flashinfer_perf.py", "test_sm90_flashinfer_local.py", "sm90_flashinfer_reference.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_flashinfer_perf.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", "@flashinfer_opensource_sidecar//:pkg", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_flashinfer_strategy",
    srcs = ["test_sm90_flashinfer_strategy.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_flashinfer_strategy.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_flashinfer_local",
    srcs = ["test_sm90_flashinfer_local.py", "sm90_flashinfer_reference.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_flashinfer_local.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", "@flashinfer_opensource_sidecar//:pkg", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_flashinfer_moe",
    srcs = ["test_sm90_flashinfer_moe.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_flashinfer_moe.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", "@flashinfer_opensource_sidecar//:pkg", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_humming_terminal_chain",
    srcs = ["test_sm90_humming_terminal_chain.py", "sm90_terminal_route_candidate.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_humming_terminal_chain.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_humming_terminal_offsets",
    srcs = ["test_sm90_humming_terminal_offsets.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_humming_terminal_offsets.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
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
    name = "test_sm90_moe_quant_scale_fused",
    srcs = ["test_sm90_moe_quant_scale_fused.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_moe_quant_scale_fused.py)"],
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
    name = "test_sm90_static_route_fusion",
    srcs = ["test_sm90_static_route_fusion.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_static_route_fusion.py)"],
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
    name = "test_sm90_humming_graph",
    srcs = ["test_sm90_humming_graph.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "2", "--cuda-major", "9", "$(location :test_sm90_humming_graph.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "2"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "2"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_dense_native_a8",
    srcs = ["test_sm90_dense_native_a8.py", "test_sm90_dense_tuning.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_dense_native_a8.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({"@//:using_cuda13_x86": [], "//conditions:default": ["@platforms//:incompatible"]}),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_humming_a8",
    srcs = ["test_sm90_humming_a8.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_humming_a8.py)"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules", "//rtp_llm:jit_cache_env", test_requirement("pytest")],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
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
    name = "test_sm90_engram_registration",
    srcs = ["test_sm90_engram_registration.py"],
    main = "test_sm90_engram_registration.py",
    deps = ["//rtp_llm/models_py:modules"],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "8"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "8"},
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

py_test(
    name = "test_sm90_humming_moe_ep",
    srcs = ["test_sm90_humming_moe_ep.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "2", "--cuda-major", "9", "$(location :test_sm90_humming_moe_ep.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "2"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "2"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_humming_moe_shapes",
    srcs = ["test_sm90_humming_moe_shapes.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_humming_moe_shapes.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

[
    py_test(
        name = name,
        srcs = [name + ".py", "//rtp_llm/test/pytest:pytest_main.py"] + (
            ["test_sm90_moe_gemm_candidates.py"] if name in [
                "test_sm90_moe_deepgemm_candidates",
                "test_sm90_moe_fp8_diagnosis",
                "test_sm90_moe_deepgemm_fp8",
                "test_sm90_moe_prefill_tuning",
                "test_sm90_moe_decode_tuning",
                "test_sm90_moe_w4a8_upstream",
                "test_sm90_moe_indexed_upstream",
                "test_sm90_moe_indexed_routes",
                "test_sm90_moe_round6",
            ] else []
        ) + (
            ["test_sm90_moe_deepgemm_candidates.py"] if name == "test_sm90_moe_deepgemm_fp8" else []
        ) + (
            ["test_sm90_moe_fp8_diagnosis.py"] if name in [
                "test_sm90_moe_prefill_tuning",
                "test_sm90_moe_decode_tuning",
            ] else []
        ),
        main = "//rtp_llm/test/pytest:pytest_main.py",
        args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :" + name + ".py)"],
        deps = [
            "//rtp_llm:testlib",
            "//rtp_llm/models_py:modules",
            "//rtp_llm:jit_cache_env",
            test_requirement("pytest"),
        ],
        tags = ["manual", "dsv41_cuda13", "H20_CU13"],
        env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
        exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
        target_compatible_with = select({
            "@//:using_cuda13_x86": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
        legacy_create_init = 0,
    )
    for name in [
        "test_sm90_moe_gemm_candidates",
        "test_sm90_moe_route_fusion",
        "test_sm90_moe_deepgemm_candidates",
        "test_sm90_moe_fp8_diagnosis",
        "test_sm90_moe_deepgemm_fp8",
        "test_sm90_moe_prefill_tuning",
        "test_sm90_moe_decode_tuning",
        "test_sm90_moe_w4a8_upstream",
        "test_sm90_moe_indexed_upstream",
        "test_sm90_moe_align",
        "test_sm90_moe_indexed_routes",
        "test_sm90_moe_round6",
        "test_sm90_dense_tuning",
    ]
]

py_test(
    name = "test_sm90_dense_g32_compare",
    srcs = [
        "test_sm90_dense_g32_compare.py",
        "test_sm90_dense_deepgemm.py",
        "test_sm90_dense_tuning.py",
        "//rtp_llm/test/pytest:pytest_main.py",
    ],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_dense_g32_compare.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_dense_deepgemm",
    srcs = [
        "test_sm90_dense_deepgemm.py",
        "test_sm90_dense_tuning.py",
        "//rtp_llm/test/pytest:pytest_main.py",
    ],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_dense_deepgemm.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_fp8_partial_probe",
    srcs = [
        "test_sm90_fp8_partial_probe.py",
        "test_sm90_dense_deepgemm.py",
        "test_sm90_dense_tuning.py",
        "//rtp_llm/test/pytest:pytest_main.py",
    ],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_fp8_partial_probe.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

py_test(
    name = "test_sm90_humming_version_regression",
    srcs = [
        "test_sm90_humming_version_regression.py",
        "test_sm90_humming_moe.py",
        "test_sm90_humming_moe_shapes.py",
        "test_sm90_mxfp8_dense.py",
        "//rtp_llm/test/pytest:pytest_main.py",
    ],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_humming_version_regression.py)"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:jit_cache_env",
        test_requirement("pytest"),
    ],
    tags = ["manual", "dsv41_cuda13", "H20_CU13"],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    legacy_create_init = 0,
)

# Run the same prenorm/Graph contracts on Hopper without changing the SM100
# CI targets or relaxing their reference tolerances.
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
    for name in ["test_v41_mhc_prenorm", "test_v41_hc_jit_warmup", "test_v41_delayed_hc_cuda_graph"]
]

_TESTS = [
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

py_test(
    name = "test_sm90_humming_moe",
    srcs = ["test_sm90_humming_moe.py", "//rtp_llm/test/pytest:pytest_main.py"],
    main = "//rtp_llm/test/pytest:pytest_main.py",
    args = ["--cuda-devices", "1", "--cuda-major", "9", "$(location :test_sm90_humming_moe.py)"],
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
