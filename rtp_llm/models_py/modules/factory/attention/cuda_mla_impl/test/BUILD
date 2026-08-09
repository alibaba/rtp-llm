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
    name = "flashinfer_mla_wrapper_test",
    srcs = ["flashinfer_mla_wrapper_test.py"],
    deps = [
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "trtllm_gen_mla_decode_op_test",
    srcs = ["trtllm_gen_mla_decode_op_test.py"],
    deps = [
        "//rtp_llm:apache-tvm-ffi",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"RTP_MLA_DECODE_KERNEL": "trtllm_gen"},
    tags = ["manual"],
    exec_properties = {'gpu': 'SM100'},
)

py_test(
    name = "tokenspeed_mla_decode_op_test",
    srcs = ["tokenspeed_mla_decode_op_test.py"],
    deps = [
        "//rtp_llm:apache-tvm-ffi",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "@tokenspeed_mla_py//:tokenspeed_mla",
    ],
    env = {"RTP_MLA_DECODE_KERNEL": "tokenspeed_mla"},
    tags = ["manual"],
    exec_properties = {'gpu': 'SM100'},
)

py_test(
    name = "paged_mla_decode_selection_test",
    srcs = ["paged_mla_decode_selection_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
