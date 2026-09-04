py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

flashinfer_deps = [
    "//rtp_llm:apache-tvm-ffi",
    "//rtp_llm:flashinfer-python",
]

flashinfer_compatible_with = select({
    "@//:using_cu12_9_or_13_x86": [],
    "@//:cuda_pre_12_9": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_library(
    name = "cp_test_utils",
    srcs = ["cp_test_utils.py"],
    deps = py_test_deps + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
)

py_test(
    name = "test_cp_utils",
    srcs = ["test_cp_utils.py"],
    deps = py_test_deps + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["A10"],
    exec_properties = {'gpu': 'A10'},
)

py_test(
    name = "test_linear_attn_utils",
    srcs = ["test_linear_attn_utils.py"],
    deps = py_test_deps + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
    target_compatible_with = flashinfer_compatible_with,
)

py_test(
    name = "test_attn_factory_cp_selection",
    srcs = ["test_attn_factory_cp_selection.py"],
    deps = py_test_deps + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
    target_compatible_with = flashinfer_compatible_with,
)

py_test(
    name = "test_allgather_cp_impl",
    srcs = ["test_allgather_cp_impl.py"],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_alltoall_cp_impl",
    srcs = ["test_alltoall_cp_impl.py"],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_allgather_overlap_impl",
    srcs = ["test_allgather_overlap_impl.py"],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
)

py_test(
    name = "test_cp_linear_attn",
    srcs = ["test_cp_linear_attn.py"],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": []
    }),
    tags = ["H20"],
    exec_properties = {'gpu': 'H20'},
    timeout = "long",
)

py_test(
    name = "test_cp_relay_distributed",
    srcs = ["test_cp_relay_distributed.py"],
    deps = py_test_deps + [":cp_test_utils"] + select({
        "@//:using_cu12_9_or_13_x86": flashinfer_deps,
        "@//:cuda_pre_12_9": flashinfer_deps,
        "//conditions:default": [],
    }),
    env = {"GPU_COUNT": "2"},
    exec_properties = {'gpu': 'H20', 'gpu_count': '2'},
    tags = ["H20"],
    timeout = "moderate",
    target_compatible_with = flashinfer_compatible_with,
)
