load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

py_test(
    name = "fia2a_plan_test",
    srcs = ["fia2a_plan_test.py"],
    data = ["//rtp_llm/models_py:fia2a_plan_sources"],
)

py_test(
    name = "fia2a_backend_test",
    srcs = ["fia2a_backend_test.py"],
    deps = ["//rtp_llm:testlib", "//rtp_llm/models_py:modules"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "2"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

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
