py_library(
    name = "moe_tp_fused_fp8",
    srcs = ["__init__.py"],
    data = [
        "fc2_reference.cuh",
        "fc2_mma.cuh",
        "fc2_staged.cuh",
        "moe_tp_fused_fp8.cu",
        "moe_tp_fused_fp8_transport.cuh",
    ],
    deps = ["//rtp_llm:torch"],
    visibility = ["//visibility:public"],
)
