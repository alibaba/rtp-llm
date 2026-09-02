_CUDA_ONLY = select({
    "@//:using_cuda": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_test(
    name = "test_config_resolver",
    srcs = ["test_config_resolver.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_cuda_strategies",
    srcs = ["test_cuda_strategies.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
    tags = ["open_skip"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_cutedsl_availability",
    srcs = ["test_cutedsl_availability.py"],
    deps = [
        "//rtp_llm:flashinfer-python",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _CUDA_ONLY,
)
