load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test = cc_test_wrapper

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "test_headers",
    hdrs = glob([
        "utils/*.h",
    ]),
)

test_deps = [
    "//rtp_llm/cpp/testing:device_test_utils",
    "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
    "//rtp_llm/cpp/cache:cache_async_context",
    "//rtp_llm/cpp/cache/test/mock:mock_allocator_lib",
    "//rtp_llm/cpp/normal_engine:normal_engine",
    "//rtp_llm/cpp/engine_base/schedulers:schedulers",
    "//rtp_llm/cpp/test:test_utils",
    "//rtp_llm/cpp/normal_engine/test:mock_engine",
    ":test_headers",
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
    "@local_config_cuda//cuda:cuda_headers",
    "@local_config_cuda//cuda:cudart",
] + torch_deps()

cc_test(
    name = "system_prompt_test",
    srcs = [
        "SystemPromptTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps,
)

cc_test(
    name = "system_prompt_constructor_test",
    srcs = [
        "SystemPromptConstructorTest.cc",
    ],
    copts = test_copts,
    data = [],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {"gpu": "A10_CU13"},
    tags = ["A10_CU13"],
    deps = test_deps + [
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
    ],
)
