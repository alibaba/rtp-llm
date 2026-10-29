# RDMA-mode cache-store tests moved from the internal tree. They link the
# closed accl-barex library and need real RDMA NICs (ACCL_USE_NICS), so
# srcs/impl deps appear only when //:using_rdma is ON (internal overlay
# builds). All targets are manual + pool-tagged: public wildcard lanes never
# schedule them; run them explicitly on an internal RDMA worker.

load("//:def.bzl", "cc_test_wrapper")
load("//bazel:device_defs.bzl", "device_impl_target")

package(default_visibility = ["//visibility:public"])

cc_test = cc_test_wrapper

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

cc_test(
    name = "cache_store_gtest",
    srcs = select({
        "//:using_rdma": [
            "main.cpp",
            "MemoryUtilTest.cpp",
            "NormalCacheStoreTest.cpp",
            "RdmaCacheStoreLoadServiceClosureTest.cpp",
            "RdmaCacheStoreServiceImplContextTest.cpp",
            "RdmaClientAndRdmaServerTest.cpp",
            "RdmaClientHostConnectionsTest.cpp",
            "RdmaConnectionTest.cpp",
            "RdmaMessagerTest.cpp",
            "RdmaServerHostConnectionsTest.cpp",
        ],
        "//conditions:default": [],
    }),
    copts = ["-fno-access-control"],
    env = {
        "CACHE_STORE_RDMA_MODE": "1",
        "DEVICE_RESERVE_MEMORY_BYTES": "128000000",
        "CACHE_STORE_MIN_REG_MEM_SIZE": "1",
        "ACCL_CONTEXT_CQE": "1024",
        "ACCL_USE_NICS": "mlx5_bond_0",
    },
    exec_properties = {"gpu": "L20_CU13"},
    tags = [
        "L20_CU13",
        "manual",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target() + cuda13_torch_link_deps + select({
        "//:using_rdma": [
            "//rtp_llm/cpp/disaggregate/cache_store/rdma:cache_store_rdma_impl",
            "//rtp_llm/cpp/disaggregate/cache_store",
            "//rtp_llm/cpp/disaggregate/cache_store/test:cache_store_test_base",
            "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
            "//aios/network/rdma:rdma_mocks",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "rdma_multi_process_gtest",
    srcs = select({
        "//:using_rdma": ["RdmaMultiProcessTest.cpp"],
        "//conditions:default": [],
    }),
    copts = ["-fno-access-control"],
    env = {
        "CACHE_STORE_RDMA_MODE": "1",
        "DEVICE_RESERVE_MEMORY_BYTES": "128000000",
        "CACHE_STORE_MIN_REG_MEM_SIZE": "1",
        "ACCL_CONTEXT_CQE": "1024",
        "ACCL_USE_NICS": "mlx5_bond_0",
    },
    exec_properties = {"gpu": "L20_CU13"},
    tags = [
        "L20_CU13",
        "manual",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + device_impl_target() + cuda13_torch_link_deps + select({
        "//:using_rdma": [
            "//rtp_llm/cpp/disaggregate/cache_store/rdma:cache_store_rdma_impl",
            "//rtp_llm/cpp/disaggregate/cache_store",
            "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        ],
        "//conditions:default": [],
    }),
)
