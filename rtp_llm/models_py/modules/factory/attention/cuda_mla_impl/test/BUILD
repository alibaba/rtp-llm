# only can run in cu129 so mark as manual
py_test(
    name = "sparse_mla_op_test",
    srcs = ["sparse_mla_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "sparse_mla_fp8_decode_op_test",
    srcs = ["sparse_mla_decode_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "flashmla_sparse_cp_op_test",
    srcs = ["flashmla_sparse_cp_op_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "mla_k_merge_non_power_two_heads_test",
    srcs = ["mla_k_merge_non_power_two_heads_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_binary(
    name = "mla_k_merge_benchmark",
    srcs = ["mla_k_merge_benchmark.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
)

py_test(
    name = "mla_page_rr_cache_test",
    srcs = ["mla_page_rr_cache_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
)

py_test(
    name = "mla_page_rr_prefix_kernel_test",
    srcs = ["mla_page_rr_prefix_kernel_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
)

py_test(
    name = "mla_page_rr_prefill_params_test",
    srcs = ["mla_page_rr_prefill_params_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual"],
)
