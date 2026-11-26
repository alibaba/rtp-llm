load("@arch_config//:arch_select.bzl", "requirement")

requirement(["numpy", "torch", "triton"])

[
    py_test(
        name = name,
        srcs = [name + ".py"],
        deps = [
            "//rtp_llm/models_py/triton_kernels:causal_conv1d",
            "//rtp_llm/models_py/triton_kernels:common",
            "//rtp_llm/models_py/triton_kernels:kda_fp32_state",
            ":numpy",
            ":torch",
            ":triton",
        ],
        env = {"GPU_COUNT": "1"},
        exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
        tags = ["manual", "SM103_CU13"],
        timeout = "long",
    )
    for name in [
        "reference_numerics_test",
        "decode_fusion_test",
    ]
]
