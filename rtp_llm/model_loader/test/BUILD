py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_stacked_moe_weight",
    srcs = ["test_stacked_moe_weight.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_pure_tp_preshard",
    srcs = ["test_pure_tp_preshard.py"],
    data = ["//rtp_llm/platforms:ppu_sources"],
    # auto_model: this test imports rtp_llm.models weight infos.
    deps = ["//rtp_llm/models_py/standalone:auto_model"],
    # CPU-only test: hides GPUs so the weight-preshard logic stays device-free.
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_compressed_w4a8_int4_per_channel",
    srcs = ["test_compressed_w4a8_int4_per_channel.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_per_channel_fp8_helpers",
    srcs = ["test_per_channel_fp8_helpers.py"],
    deps = py_test_deps,
)

# Verified on L20_CU13 (sm89) after the sm8x lane moved to L20: the
# per-channel suite runs 30/30 with 0 skips, including
# test_fp8_still_runs_the_device_conversion, so the device FP8 cast is fine
# on this pool.
py_test(
    name = "test_compressed_w8a8_int8_per_channel",
    srcs = ["test_compressed_w8a8_int8_per_channel.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_swizzle_load_config",
    srcs = ["test_swizzle_load_config.py"],
    deps = py_test_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_model_weight_info",
    srcs = ["test_model_weight_info.py"],
    deps = py_test_deps + [
        # auto_model: the finalize coverage imports rtp_llm.models.base_model.
        "//rtp_llm/models_py/standalone:auto_model",
    ],
    exec_properties = {"gpu": "L20_CU13"},
    # Metadata-only test; L20_CU13 supplies the runtime without an H20 reservation.
    tags = ["L20_CU13"],
)

py_test(
    name = "test_inline_fp8_quant",
    srcs = ["test_inline_fp8_quant.py"],
    deps = py_test_deps,
    tags = ["rocm"],
    exec_properties = {
        'gpu': 'MI308X-ROCM7',
    },
)
