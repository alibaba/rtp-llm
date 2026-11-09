package(default_visibility = ["//visibility:public"])

py_test(
    name = "residual_up_mix_test",
    srcs = ["residual_up_mix_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/triton_kernels/gated_residual/test:residual_fixtures"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_norm_fp8_test",
    srcs = ["residual_norm_fp8_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/triton_kernels/gated_residual/test:residual_fixtures"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_medium_up_mix_test",
    srcs = ["residual_medium_up_mix_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/triton_kernels/gated_residual/test:residual_fixtures"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = 'residual_prefill_inject_norm_test',
    srcs = ["residual_prefill_inject_norm_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/triton_kernels/gated_residual/test:residual_fixtures"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_inject_projection_test",
    srcs = ["residual_inject_projection_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/triton_kernels/gated_residual/test:residual_fixtures"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = 'residual_splitk_inject_test',
    srcs = ["residual_splitk_inject_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/triton_kernels/gated_residual/test:residual_fixtures"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_test(
    name = "residual_prefill_up_mix_test",
    srcs = ["residual_prefill_up_mix_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py/triton_kernels/gated_residual/test:residual_fixtures"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)

py_library(
    name = "residual_fixtures",
    srcs = ["residual_fixtures.py"],
    deps = ["//rtp_llm:testlib"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "fp8_dense_gemv_test",
    srcs = ["fp8_dense_gemv_test.py"],
    deps = ["//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": "1"},
)
