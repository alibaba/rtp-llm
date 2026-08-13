load("//bazel:arch_select.bzl", "tokenspeed_mla_test_deps")

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
    name = "flashinfer_decode_params_test",
    srcs = ["flashinfer_decode_params_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "tokenspeed_mla_impl_test",
    srcs = ["tokenspeed_mla_impl_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "tokenspeed_mla_decode_op_test",
    srcs = ["tokenspeed_mla_decode_op_test.py"],
    deps = [
        "//rtp_llm:apache-tvm-ffi",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ] + tokenspeed_mla_test_deps(),
    env = {"RTP_MLA_DECODE_KERNEL": "tokenspeed_mla"},
    tags = ["manual"],
    exec_properties = {'gpu': 'SM100'},
)
