load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "custom_all_gather_test",
    srcs = ["test_custom_all_gather.py"],
    main = "test_custom_all_gather.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
)

py_test(
    name = "custom_all_gather_gpu_test",
    srcs = ["test_custom_all_gather_gpu.py"],
    main = "test_custom_all_gather_gpu.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    timeout = "long",
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300", "gpu_count": "8"},
)

py_test(
    name = "moe_residual_gpu_test",
    srcs = ["test_moe_residual_gpu.py"],
    main = "test_moe_residual_gpu.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1", "DSV4_MOE_STRICT_FUSED": "0"},
    timeout = "short",
    tags = ["manual", "H20"],
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
)

py_test(
    name = "moe_input_packer_gpu_test",
    srcs = ["test_moe_input_packer_gpu.py"],
    main = "test_moe_input_packer_gpu.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1", "DSV4_MOE_STRICT_FUSED": "0"},
    timeout = "short",
    tags = ["manual", "H20"],
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
)

py_test(
    name = "producers_gpu_test",
    srcs = ["test_producers_gpu.py"],
    main = "test_producers_gpu.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300", "gpu_count": "1"},
    visibility = ["//visibility:public"],
)

py_test(
    name = "mtp_gpu_test",
    srcs = ["test_mtp_gpu.py"],
    main = "test_mtp_gpu.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300", "gpu_count": "1"},
)

py_test(
    name = "projection_ktp_gpu_test",
    srcs = ["test_projection_ktp_gpu.py"],
    main = "test_projection_ktp_gpu.py",
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "H20"],
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
    visibility = ["//visibility:public"],
)

[
    py_test(
        name = "projection_ktp_nccl_%d_test" % ranks,
        srcs = ["test_projection_ktp_nccl.py"],
        main = "test_projection_ktp_nccl.py",
        deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
        env = {"GPU_COUNT": str(ranks)},
        exec_properties = {"gpu": "H20", "gpu_count": str(ranks)},
        tags = ["manual", "H20"],
        timeout = "long",
    )
    for ranks in [1, 2, 4, 8]
]

py_test(
    name = "fp8_decode_ag_gpu_test",
    srcs = ["test_fp8_decode_ag_gpu.py"],
    main = "test_fp8_decode_ag_gpu.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    env = {"GPU_COUNT": "8"},
    exec_properties = {"gpu": "B300", "gpu_count": "8"},
    tags = ["manual", "B300"],
    timeout = "long",
)

py_test(
    name = "chunkwise_rdma_publisher_test",
    srcs = ["test_chunkwise_rdma_publisher.py"],
    main = "test_chunkwise_rdma_publisher.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

py_test(
    name = "projection_ktp_test",
    srcs = ["test_projection_ktp.py"],
    main = "test_projection_ktp.py",
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:testlib",
    ],
    timeout = "short",
)

py_test(
    name = "push_reduce_scatter_test",
    srcs = ["test_push_reduce_scatter.py"],
    main = "test_push_reduce_scatter.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300"},
)

py_test(
    name = "push_reduce_scatter_gpu_test",
    srcs = ["test_push_reduce_scatter_gpu.py"],
    main = "test_push_reduce_scatter_gpu.py",
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
    timeout = "long",
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300", "gpu_count": "8"},
)

# TP8 retains the existing target names above. TP16 can be run using torchrun
# across hosts in one NVLink fabric; these targets exercise a single host.
[
    py_test(
        name = "%s_tp%d_gpu_test" % (operator, tp),
        srcs = ["test_%s_gpu.py" % operator],
        main = "test_%s_gpu.py" % operator,
        args = ["--world-size", str(tp)],
        deps = ["//rtp_llm/models_py:models", "//rtp_llm:testlib"],
        timeout = "long",
        tags = ["manual", "B300"],
        exec_properties = {"gpu": "B300", "gpu_count": str(tp)},
    )
    for operator in ["custom_all_gather", "push_reduce_scatter"]
    for tp in [2, 4]
]
