package(default_visibility = ["//visibility:public"])

cc_library(
    name = "remote_mock_types",
    hdrs = ["RemoteMockTypes.h"],
    deps = [
        "//rtp_llm/cpp/disaggregate/cache_store/remote_mock:remote_mock_core",
    ],
)

# Compile with the existing GroupSet/transfer library to avoid a factory/adapter cycle.
filegroup(name = "adapter_srcs", srcs = ["RemoteMockAdapter.cc", "RemoteMockWriteback.cc"])
filegroup(name = "adapter_hdrs", srcs = ["RemoteMockAdapter.h", "RemoteMockWriteback.h"])
alias(name = "remote_mock_adapter", actual = "//rtp_llm/cpp/cache/block_tree_cache")
