load("@arch_config//:arch_select.bzl", "requirement")

requirement(["numpy", "torch", "triton"])

[
    py_test(
        name = name,
        srcs = [name + ".py"],
        deps = [
            "//rtp_llm/models_py/triton_kernels:causal_conv1d",
            "//rtp_llm/models_py/triton_kernels:common",
            "//rtp_llm/models_py/triton_kernels:glm53_kda",
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
        "glm53_replay_test",
        "glm53_reference_numerics_test",
        "glm53_decode_fusion_test",
    ]
]
