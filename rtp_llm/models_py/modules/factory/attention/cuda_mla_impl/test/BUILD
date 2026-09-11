# only can run in cu129 so mark as manual
load("//bazel:arch_select.bzl", "requirement")

requirement(["torch", "triton", "flash-mla"])

py_test(
    name = "pinned_mla_cache_test",
    srcs = ["pinned_mla_cache_test.py"],
    deps = [
        "//rtp_llm/models_py:pinned_mla_cache",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        ":torch",
        ":triton",
        ":flash-mla",
    ],
    tags = ["manual"],
    exec_properties = {"gpu": "L20D_DEV"},
)

py_test(
    name = "glm53_prefill_workspace_test",
    srcs = ["glm53_prefill_workspace_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

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
    main = "sparse_mla_decode_op_test.py",
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

py_test(
    name = "hy4_tiered_cache_test",
    srcs = ["hy4_tiered_cache_test.py"],
    deps = [
        "//rtp_llm/models_py:pinned_mla_cache",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        ":torch",
        ":triton",
    ],
    tags = ["manual"],
)
