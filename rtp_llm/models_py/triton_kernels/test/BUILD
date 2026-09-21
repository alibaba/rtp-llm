load("//bazel:arch_select.bzl", "requirement")

requirement(["numpy", "packaging", "torch", "triton"])

py_test(
    name = "address_overflow_test",
    srcs = ["address_overflow_test.py"],
    env = {"TRITON_INTERPRET": "1"},
    deps = [
        "//rtp_llm/models_py:dsv4_fused_kernel_sources",
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ],
)

py_test(
    name = "address_overflow_gpu_test",
    srcs = ["address_overflow_gpu_test.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu_count": "1"},
    tags = ["manual"],  # Actual >2^31-element storage needs at least 9 GiB free.
    deps = [
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":packaging",
        ":torch",
        ":triton",
    ],
)
