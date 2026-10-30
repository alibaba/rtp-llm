load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "cc_test_wrapper")
load("//bazel:device_defs.bzl", "device_impl_target")

cc_test(
    name = "deferred_read_queue_test",
    srcs = ["DeferredReadQueueTest.cc"],
    deps = [
        "//rtp_llm/cpp/rdma_transport:deferred_read_queue",
        "@com_google_googletest//:gtest_main",
    ],
)

# Tensor RDMA smoke test moved from the internal tree. It links the closed
# accl-barex library, so srcs and the impl dep appear only when
# //:using_rdma is ON (internal overlay builds). manual keeps public
# wildcard lanes from scheduling it; the internal smoke suite runs it via
# its explicit test_suite member list, which includes manual targets.
cc_test_wrapper(
    name = "barex_rdma_transport_smoke_test",
    size = "medium",
    timeout = "moderate",
    srcs = select({
        "//:using_rdma": [
            "BarexRdmaTransportSmokeTest.cc",
            "test_main.cc",
        ],
        "//conditions:default": [],
    }),
    env = {
        "ARPC_RDMA_MEMPOOL_ENABLE_AUTO_SHRINK": "false",
        "ARPC_RDMA_MEMPOOL_GPU_MAX_TOTAL_BYTES": "268435456",
        "ARPC_RDMA_RESERVE_GPU_MEM": "",
        "GPU_COUNT": "2",
        "LD_PRELOAD": "libtorch_cpu.so",
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {
        "gpu": "H20_CU13",
        "gpu_count": "2",
    },
    tags = [
        "H20_CU13",
        "manual",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target() + torch_deps() + select({
        "//:using_rdma": [
            "//rtp_llm/cpp/rdma_transport/barex:barex_rdma_transport_impl",
            "@havenask//aios/autil:log",
            "//rtp_llm/cpp/metrics",
            "//rtp_llm/cpp/multimodal_processor/transport/rdma:mm_rdma_reader",
        ],
        "//conditions:default": [],
    }),
)
