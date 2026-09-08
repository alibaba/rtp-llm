py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test(
    name = "test_stacked_moe_weight",
    srcs = ["test_stacked_moe_weight.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_pure_tp_preshard",
    srcs = ["test_pure_tp_preshard.py"],
    # auto_model: this test imports rtp_llm.models weight infos.
    deps = ["//rtp_llm/models_py/standalone:auto_model"],
    # Accelerator hooks, FP8 device copies, and handle recycling need a GPU target.
    env = {"CUDA_VISIBLE_DEVICES": ""},
)

py_test(
    name = "test_compressed_w4a8_int4_per_channel",
    srcs = ["test_compressed_w4a8_int4_per_channel.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_per_channel_fp8_helpers",
    srcs = ["test_per_channel_fp8_helpers.py"],
    deps = py_test_deps,
)

# Verified on sm86: 19/19, 1/1 and 30/30 pass with 0 skips, including
# test_fp8_still_runs_the_device_conversion, so the device FP8 cast is fine there.
py_test(
    name = "test_compressed_w8a8_int8_per_channel",
    srcs = ["test_compressed_w8a8_int8_per_channel.py"],
    deps = py_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "test_model_weight_info",
    srcs = ["test_model_weight_info.py"],
    deps = py_test_deps + [
        # auto_model: the finalize coverage imports rtp_llm.models.base_model.
        "//rtp_llm/models_py/standalone:auto_model",
    ],
    # Metadata-only weight-info test (fake ckpt objects, no device copies):
    # no H20 pin, runs on the default gpu_cuda12 pool.
    exec_properties = {'gpu':'A10'},
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

py_test(
    name = "load_embedding_weight_test",
    srcs = ["load_embedding_weight_test.py"],
    deps = py_test_deps + ["//rtp_llm/models_py/standalone:auto_model"],
    exec_properties = {"gpu": "A10"},
)
