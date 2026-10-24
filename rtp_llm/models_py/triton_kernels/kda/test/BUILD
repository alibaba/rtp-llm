load("@arch_config//:arch_select.bzl", "requirement")
load("@pip_gpu_cuda13_torch//:requirements.bzl", cuda13_requirement = "requirement")
load("@pip_cuda13_arm_torch//:requirements.bzl", cuda13_arm_requirement = "requirement")

requirement([
    "triton",
    "torch",
    "numpy",
    "einops",
    "packaging",
])

py_test(
    name = "test_kda_ops",
    srcs = [
        "test_kda_ops.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda",
    ] + [
        ":einops",
        ":numpy",
        ":packaging",
        ":torch",
        ":triton",
    ] + select({
        "@//:using_cuda13_x86": [cuda13_requirement("cuda-linear-attention")],
        "@//:using_cuda13_arm": [cuda13_arm_requirement("cuda-linear-attention")],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "test_chunk_prefill",
    srcs = ["test_chunk_prefill.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm/models_py/triton_kernels:kda",
        ":torch",
        ":triton",
        ":einops",
        ":numpy",
        ":packaging",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)
