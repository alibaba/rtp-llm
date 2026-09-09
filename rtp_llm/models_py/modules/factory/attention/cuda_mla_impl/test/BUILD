py_test(
    name = "trtllm_kv_compat_test",
    srcs = ["trtllm_kv_compat_test.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:triton",
        "//rtp_llm/models_py/triton_kernels:sparse_mla",
    ],
    tags = ["manual"],
)

py_test(
    name = "trtllm_sparse_integration_test",
    srcs = ["trtllm_sparse_integration_test.py"],
    # Tests compile production ASTs without importing CUDA/service extensions.
    data = ["//rtp_llm/models_py:modules"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "trtllm_sparse_decode_test",
    srcs = ["trtllm_sparse_decode_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:flashinfer-cubin",
    ],
    tags = ["manual"],
)

py_test(
    name = "trtllm_sparse_cmp_test",
    srcs = [
        "trtllm_sparse_cmp_test.py",
        "trtllm_sparse_decode_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:flashinfer-cubin",
    ],
    tags = ["manual"],
)

py_binary(
    name = "glm53_trtllm_compat_benchmark",
    srcs = [
        "glm53_trtllm_compat_benchmark.py",
        "glm53_attention_unit_benchmark.py",
        "glm53_attention_unit_paths.py",
    ],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:flashinfer-cubin",
    ],
    tags = ["manual"],
)

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
