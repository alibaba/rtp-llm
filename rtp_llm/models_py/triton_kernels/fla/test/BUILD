load("@arch_config//:arch_select.bzl", "requirement")

requirement(["triton", "torch", "numpy", "einops", "packaging"])

py_test(
    name = "test_gdn_block_prefill",
    srcs = [
        "test_gdn_block_prefill.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_gdn_decode",
    srcs = [
        "test_gdn_decode.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_chunk_prefill",
    srcs = [
        "test_chunk_prefill.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)

# Reuse test_chunk_prefill on AMD to verify chunk-gdn Triton path on ROCm.
# NOTE: Gluon fwd_h kernel requires gfx950 (MI355X/CDNA4). On MI308X (gfx942)
# this exercises the Triton fallback only. When CI adds gfx950 machines,
# the Gluon path will be auto-dispatched via _is_gluon_beneficial.
py_test(
    name = "test_chunk_prefill_amd",
    srcs = [
        "test_chunk_prefill.py"
    ],
    main = "test_chunk_prefill.py",
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test(
    name = "test_l2norm",
    srcs = [
        "test_l2norm.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)