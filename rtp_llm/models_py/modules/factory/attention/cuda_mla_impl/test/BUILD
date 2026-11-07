load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

py_test(
    name = "fp8_prefill_dequant_metadata_test",
    srcs = ["fp8_prefill_dequant_metadata_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    env = {"GPU_COUNT": "0"},
)

[
    py_test(
        name = "dsa_replicated_query_tp%d_test" % tp,
        srcs = ["dsa_replicated_query_test.py"],
        main = "dsa_replicated_query_test.py",
        deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
        env = {"GPU_COUNT": str(tp)},
        tags = ["SM103_CU13", "manual"],
        exec_properties = {"gpu": "SM103_CU13", "gpu_count": str(tp)},
        timeout = "long",
    )
    for tp in [2, 8]
]

# only can run in cu129 so mark as manual
py_test(
    name = "sparse_mla_op_test",
    srcs = ["sparse_mla_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "sparse_mla_fp8_decode_op_test",
    srcs = ["sparse_mla_decode_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

py_test(
    name = "flashmla_sparse_cp_op_test",
    srcs = ["flashmla_sparse_cp_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = [
        "H20_CU13",
        "manual",
    ],
    exec_properties = {"gpu": "H20_CU13"},
)

cuda13_sm100_py_test(
    name = "mla_dcp_kernel_contract_test",
    srcs = ["mla_dcp_kernel_contract_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

cuda13_sm100_py_test(
    name = "tokenspeed_mla_page_rr_test",
    srcs = ["tokenspeed_mla_page_rr_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "page_rr_jit_warmup_test",
    srcs = ["page_rr_jit_warmup_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/triton_kernels:mla",
        "//rtp_llm:triton",
    ],
    tags = ["L20_CU13"],
    exec_properties = {"gpu": "L20_CU13"},
    target_compatible_with = select({
        "//:using_cuda": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cuda13_sm100_py_test(
    name = "dsa_decode_positions_lifetime_test",
    srcs = ["dsa_decode_positions_lifetime_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)
