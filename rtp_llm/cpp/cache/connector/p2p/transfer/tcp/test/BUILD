load("//:def.bzl", "copts")

load("@arch_config//:arch_select.bzl", "torch_deps")

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

test_copts = [
    "-fno-access-control",
] + copts()

cc_test(
    name = "tcp_task_context_test",
    srcs = [
        "TcpTaskContextTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'H20'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp:tcp_task_context",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/models_py/bindings/core:runtime_ops_test_impls",
        "//rtp_llm/cpp/metrics:metrics",
        "@havenask//aios/autil:log",
        "@com_google_googletest//:gtest",
    ] + torch_deps() + cuda13_torch_link_deps,
)

cc_test(
    name = "tcp_transfer_service_test",
    srcs = [
        "TcpTransferServiceTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'H20'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp:tcp_transfer_service",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/models_py/bindings/core:runtime_ops_test_impls",
        "//rtp_llm/cpp/metrics:metrics",
        "@havenask//aios/autil:log",
        "@com_google_googletest//:gtest",
    ] + torch_deps() + cuda13_torch_link_deps,
)

cc_test(
    name = "tcp_sender_receiver_test",
    srcs = [
        "TcpSenderReceiverTest.cc",
        "main.cc",
    ],
    copts = test_copts,
    exec_properties = {'gpu':'H20'},
    deps = [
        "//rtp_llm/cpp/cache/connector/p2p/transfer/tcp:tcp_backend",
        "//rtp_llm/cpp/cache/connector/p2p/transfer:basic_types",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/runtime:cuda_runtime_hdrs",
        "//rtp_llm/cpp/utils:time_util",
        "//rtp_llm/models_py/bindings/core:runtime_ops_test_impls",
        "//rtp_llm/cpp/metrics:metrics",
        "@havenask//aios/autil:log",
        "@com_google_googletest//:gtest",
    ] + torch_deps() + cuda13_torch_link_deps,
)
