load("//bazel:arch_select.bzl", "requirement")

requirement(["torch", "triton"])

py_test(
    name = "cached_launch_compat_test",
    srcs = ["cached_launch_compat_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:common",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "cached_launch_test",
    srcs = ["cached_launch_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:common",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
    tags = ["manual", "H20"],
)

py_test(
    name = "fp8_scale_layout_test",
    srcs = ["fp8_scale_layout_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:common",
        ":torch",
        ":triton",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "B300", "gpu_count": "1"},
    tags = ["manual", "B300"],
)

py_test (
    name = "silu_mul_masked_test",
    srcs = ["silu_mul_masked_test.py"],
    deps = [
        "//rtp_llm:matplotlib",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu':'H20', 'gpu_count':'1'},
)

py_test(
    name = "test_moe_gating",
    srcs = ["test_moe_gating.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["H20"],
    exec_properties = {'gpu': 'H20', 'gpu_count': '1'},
)
