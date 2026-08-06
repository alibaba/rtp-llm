load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "telemetry_test_deps", "torch_deps")
load("//bazel:device_defs.bzl",
    "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

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

py_test (
    name = "model_rpc_client_test",
    srcs = ["model_rpc_client_test.py"],
    deps = [
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util",
    ] + telemetry_test_deps(),
    data = [
        "//:th_transformer"
    ],
    exec_properties = {'gpu':'H20'},
)

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/cpp/models:models",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    "//rtp_llm/cpp/model_rpc:model_rpc_server",
    "//rtp_llm/cpp/normal_engine:normal_engine",
    "//rtp_llm/cpp/testing:test_log_capture",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@havenask//aios/autil:base64",
    "@havenask//aios/autil:zlib",
] + torch_deps() + device_impl_target() + cuda13_torch_link_deps

# cc_test(
#     name = "model_rpc_server_test",
#     srcs = [
#         "RpcServerTest.cc",
#     ],
#     copts = test_copts,
#     deps = [
#         "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
#         "//rtp_llm/cpp:components",
#         "@com_google_googletest//:gtest",
#         "@com_google_googletest//:gtest_main",
#         "//rtp_llm/cpp/model_rpc:model_rpc_server",
#         "//rtp_llm/cpp/utils:utils",
#     ] + torch_deps(),
# )

cc_test(
    name = "local_rpc_server_test",
    srcs = ["LocalRpcServerTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "query_converter_test",
    srcs = [
        "QueryConverterTest.cc",
    ],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/models:logits_processor",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "broadcast_manager_test",
    srcs = [
        "BroadcastManagerTest.cc",
    ],
    data = [],
    copts = test_copts,
    deps = test_deps + [
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
        "@havenask//aios/autil:net",
    ],
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "prefill_rpc_server_test",
    srcs = ["PrefillRpcServerTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "decode_rpc_server_test",
    srcs = ["DecodeRpcServerTest.cc"],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "recent_cache_key_window_test",
    srcs = [
        "RecentCacheKeyWindowTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:recent_cache_key_window",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + cuda13_torch_link_deps,
)

cc_test(
    name = "response_buffer_test",
    srcs = [
        "ResponseBufferTest.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "prefill_batch_rpc_server_test",
    srcs = [
        "PrefillBatchRpcServerTest.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "rpc_writer_cancellation_test",
    srcs = [
        "RpcWriterCancellationTest.cc",
    ],
    copts = test_copts,
    deps = test_deps + [
        "@io_opentelemetry_cpp//exporters/memory:in_memory_span_exporter",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

cc_test(
    name = "rpc_server_runtime_meta_test",
    srcs = [
        "RpcServerRuntimeMetaTest.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)
