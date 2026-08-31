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
    name = "glm53_nope_sparse_mla_test",
    srcs = ["sparse_mla_op_test.py"],
    main = "sparse_mla_op_test.py",
    args = ["SparseMlaOpTest.test_nope_prefill_cache_write_then_decode"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {
        'gpu': 'L20A_DEV',
        'gpu_count': '1',
    },
)

py_test(
    name = "glm53_nope_fp8_sparse_mla_test",
    srcs = ["sparse_mla_op_test.py"],
    main = "sparse_mla_op_test.py",
    args = ["SparseMlaOpTest.test_nope_fp8_cache_write_then_sparse_attention_page128"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {
        'gpu': 'L20A_DEV',
        'gpu_count': '1',
    },
)

py_test(
    name = "glm53_grouped_topk_test",
    srcs = ["sparse_mla_op_test.py"],
    main = "sparse_mla_op_test.py",
    args = ["SparseMlaOpTest.test_glm53_grouped_topk_expansion"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {
        'gpu': 'L20A_DEV',
        'gpu_count': '1',
    },
)

py_test(
    name = "glm53_topk2051_kernel_test",
    srcs = ["sparse_mla_op_test.py"],
    main = "sparse_mla_op_test.py",
    args = ["SparseMlaOpTest.test_glm53_topk2051_prefill_and_decode_kernel"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {
        'gpu': 'L20A_DEV',
        'gpu_count': '1',
    },
)

py_test(
    name = "glm53_sparse_mla_registry_test",
    srcs = ["sparse_mla_registry_test.py"],
    main = "sparse_mla_registry_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {
        'gpu': 'L20A_DEV',
        'gpu_count': '1',
    },
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
    name = "test_target_verify_kernel",
    srcs = ["test_target_verify_kernel.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_plan_fp8_no_sync",
    srcs = ["test_plan_fp8_no_sync.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_prefill_cuda_graph_kernel",
    srcs = ["test_prefill_cuda_graph_kernel.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    tags = ["manual"],
    exec_properties = {'gpu': 'H20'},
)
