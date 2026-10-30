py_test(
    name = "page_rr_fia2a_test",
    srcs = ["page_rr_fia2a_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:triton",
        "//rtp_llm:nvidia-cutlass-dsl",
    ],
    env = {"GPU_COUNT": "4", "FIA2A_EXPECTED_CC": "10.3"},
    tags = ["SM103_CU13", "multi_device"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "4"},
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "page_rr_fia2a_test_arm",
    srcs = ["page_rr_fia2a_test.py"],
    main = "page_rr_fia2a_test.py",
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm:triton",
        "//rtp_llm:nvidia-cutlass-dsl",
    ],
    env = {"GPU_COUNT": "4", "FIA2A_EXPECTED_CC": "10.0"},
    tags = ["SM100_ARM_CU13", "multi_device"],
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "4"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "attn_factory_generation_prefill_cuda_graph_test",
    srcs = ["attn_factory_generation_prefill_cuda_graph_test.py"],
    tags = ["H20_CU13"],
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "1",
    },
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "attn_factory_generation_prefill_cuda_graph_test_rocm",
    main = "attn_factory_generation_prefill_cuda_graph_test.py",
    srcs = ["attn_factory_generation_prefill_cuda_graph_test.py"],
    tags = ["rocm"],
    exec_properties = {"gpu": "MI308X-ROCM7"},
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)
