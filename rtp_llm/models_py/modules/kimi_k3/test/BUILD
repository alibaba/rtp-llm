load("@rules_python//python:defs.bzl", "py_test")

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
