package(default_visibility = ["//rtp_llm/cpp/disaggregate/cache_store:__subpackages__"])

cc_binary(
    name = "cache_store_client",
    srcs = [
        "CacheStoreClient.cpp",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "//rtp_llm/cpp/disaggregate/cache_store/test:cache_store_test_base",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    ],
    copts = ["-fno-access-control",],
)

cc_binary(
    name = "cache_store_server",
    srcs = [
        "CacheStoreServer.cpp",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "//rtp_llm/cpp/disaggregate/cache_store/test:cache_store_test_base",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
    ]
)


# The RDMA perf binaries need the accl-barex backend, which exists only under
# //:using_rdma (internal deps overlay, see ../rdma/BUILD); without it they
# compile to a stub that explains the requirement.
cc_binary(
    name = "rdma_client",
    srcs = select({
        "//:using_rdma": ["RdmaClientMain.cpp"],
        "//conditions:default": ["RdmaUnavailableMain.cpp"],
    }),
    deps = select({
        "//:using_rdma": [
            "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
            "//rtp_llm/cpp/disaggregate/cache_store/rdma:cache_store_rdma_impl",
            "@havenask//aios/autil:lock_free",
            "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        ],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "rdma_server",
    srcs = select({
        "//:using_rdma": ["RdmaServerMain.cpp"],
        "//conditions:default": ["RdmaUnavailableMain.cpp"],
    }),
    deps = select({
        "//:using_rdma": [
            "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
            "//rtp_llm/cpp/disaggregate/cache_store/rdma:cache_store_rdma_impl",
            "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        ],
        "//conditions:default": [],
    }),
)