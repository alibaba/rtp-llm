py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

py_library(
    name = "cp_test_utils",
    srcs = ["cp_test_utils.py"],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_cp_utils",
    srcs = ["test_cp_utils.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_allgather_cp_impl",
    srcs = ["test_allgather_cp_impl.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_alltoall_cp_impl",
    srcs = ["test_alltoall_cp_impl.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_allgather_overlap_impl",
    srcs = ["test_allgather_overlap_impl.py"],
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
)

py_test(
    name = "test_cp_linear_attn",
    srcs = ["test_cp_linear_attn.py"],
    exec_properties = {"gpu": "H20_CU13"},
    tags = [
        "H20_CU13",
        "manual",
    ],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cuda12_9_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
)
