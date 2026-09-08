load("//bazel:arch_select.bzl", "tokenspeed_mla_test_deps")

py_test(
    name = "flashmla_forward_plan_test",
    srcs = ["flashmla_forward_plan_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

py_test(
    name = "flashmla_state_merge_test",
    srcs = ["flashmla_state_merge_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["B300"],
    exec_properties = {'gpu': 'B300'},
)

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
    name = "flashmla_dense_prefill_params_test",
    srcs = ["flashmla_dense_prefill_params_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
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
    tags = ["manual"],
    exec_properties = {'gpu': 'SM100'},
)

py_test(
    name = "flashmla_dense_prefill_packed_kv_test",
    srcs = [
        "flashmla_dense_prefill_packed_kv_test.py",
        "flashmla_forward_test_utils.py",
    ],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["B300"],
    exec_properties = {'gpu': 'B300'},
)

py_test(
    name = "page_rr_mla_decode_test",
    srcs = ["page_rr_mla_decode_test.py", "flashmla_forward_test_utils.py"],
    timeout = "long",
    data = ["//rtp_llm/cpp/cuda_graph/tests:test_cuda_graph_runner"],
    deps = [
        "//rtp_llm:apache-tvm-ffi",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/distributed:collective_torch",
    ] + tokenspeed_mla_test_deps(),
    tags = ["manual"],
    exec_properties = {'gpu': 'SM100', 'gpu_count': '8'},
)

py_binary(
    name = "flashmla_dense_prefill_packed_kv_benchmark",
    srcs = ["flashmla_dense_prefill_packed_kv_benchmark.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual", "B300"],
    exec_properties = {'gpu': 'B300'},
)

py_test(
    name = "mla_fp8_test",
    srcs = ["mla_fp8_test.py", "flashmla_forward_test_utils.py", "flashmla_dense_prefill_packed_kv_test.py"],
    deps = [
        "//rtp_llm:apache-tvm-ffi",
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ] + tokenspeed_mla_test_deps(),
    tags = ["manual", "SM100"],
    exec_properties = {'gpu': 'SM100'},
)
