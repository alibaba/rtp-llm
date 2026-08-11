load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")
load("//bazel:device_defs.bzl",
    "device_impl_target", "device_test_envs")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

py_test (
    name = "model_rpc_client_test",
    srcs = ["model_rpc_client_test.py"],
    deps = [
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util",
    ],
    data = [
        "//:th_transformer"
    ],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "aux_info_wire_test",
    srcs = ["aux_info_wire_test.py"],
    data = [":aux_info_wire_fixture"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto",
    ],
)

cc_binary(
    name = "aux_info_wire_fixture",
    srcs = ["AuxInfoWireFixture.cc"],
    deps = [
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    ],
)

cc_test(
    name = "prefill_peer_selector_test",
    srcs = ["PrefillPeerSelectorTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:prefill_peer_selector",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "prefill_run_waiter_test",
    srcs = ["PrefillRunWaiterTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:prefill_run_waiter",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rpc_fanout_utils_test",
    srcs = ["RpcFanoutUtilsTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:rpc_fanout_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "remote_load_budget_test",
    srcs = ["RemoteLoadBudgetTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:propagated_client_context",
        "//rtp_llm/cpp/model_rpc:remote_load_budget",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "request_deadline_budget_test",
    srcs = ["RequestDeadlineBudgetTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:request_deadline_budget",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rpc_error_code_test",
    srcs = ["RpcErrorCodeTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:model_rpc_server",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@havenask//aios/autil:base64",
        "@havenask//aios/autil:zlib",
    ],
)

cc_test(
    name = "cache_transfer_block_selector_test",
    srcs = ["CacheTransferBlockSelectorTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:cache_transfer_block_selector",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cache_transfer_block_selector_allocation_test",
    srcs = ["CacheTransferBlockSelectorAllocationTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/cache:kv_cache_group",
        "//rtp_llm/cpp/cache/test:block_pool_test_helper",
        "//rtp_llm/cpp/model_rpc:cache_transfer_block_selector",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cache_transfer_lease_test",
    srcs = ["CacheTransferLeaseTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:cache_transfer_lease",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "remote_load_fence_test",
    srcs = ["RemoteLoadFenceTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:remote_load_fence",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "remote_load_lease_retainer_test",
    srcs = ["RemoteLoadLeaseRetainerTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:remote_load_lease_retainer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "remote_load_lease_protocol_test",
    srcs = ["RemoteLoadLeaseProtocolTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:cache_transfer_lease",
        "//rtp_llm/cpp/model_rpc:remote_load_fence",
        "//rtp_llm/cpp/model_rpc:remote_load_lease_retainer",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "remote_load_target_tracker_test",
    srcs = ["RemoteLoadTargetTrackerTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:remote_load_target_tracker",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "sampler_generator_state_test",
    srcs = ["SamplerGeneratorStateTest.cc"],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/model_rpc:sampler_generator_state",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
)

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/cpp/models:models",
    "//rtp_llm/cpp/config:config_modules",
    "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
    "//rtp_llm/cpp/model_rpc:model_rpc_server",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@havenask//aios/autil:base64",
    "@havenask//aios/autil:zlib",
] + torch_deps() + device_impl_target()

cc_test(
    name = "remote_load_admission_test",
    srcs = ["RemoteLoadAdmissionTest.cc"],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "request_admission_service_test",
    srcs = ["RequestAdmissionServiceTest.cc"],
    copts = test_copts,
    deps = test_deps,
)

cc_test(
    name = "local_rpc_output_cancellation_test",
    srcs = ["LocalRpcOutputCancellationTest.cc"],
    copts = test_copts,
    deps = test_deps,
)

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
    name = "query_converter_test",
    srcs = [
        "QueryConverterTest.cc",
    ],
    copts = test_copts,
    deps = test_deps,
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
    name = "local_rpc_server_outcome_test",
    srcs = [
        "LocalRpcServerOutcomeTest.cc",
    ],
    copts = test_copts,
    deps = test_deps,
    env = {},
    exec_properties = {'gpu':'H20'},
)
