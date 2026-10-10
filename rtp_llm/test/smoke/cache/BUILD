load("//:def.bzl", "copts")

load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "cache_store_smoke_rank",
    testonly = True,
    srcs = ["CacheStoreSmokeRank.cc", "CacheStoreSmokeEndpoint.h", "CacheSmokeSupport.h"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/cache/connector/p2p:layer_cache_buffer",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
)

sh_test(
    name = "cache_store_mha_bf16_1to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_bf16_1to1_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_bf16_1to1_1k.json"],
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mha_bf16_1to2_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_bf16_1to2_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_bf16_1to2_1k.json"],
    env = {"GPU_COUNT": "3"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mha_bf16_2to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_bf16_2to1_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_bf16_2to1_1k.json"],
    env = {"GPU_COUNT": "3"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mha_bf16_2to2_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_bf16_2to2_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_bf16_2to2_1k.json"],
    env = {"GPU_COUNT": "4"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mha_int8_1to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_int8_1to1_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_int8_1to1_1k.json"],
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

test_suite(
    name = "cache_store_reachable_mha_smoke",
    tests = [
        ":cache_store_mha_bf16_1to1_1k_smoke",
        ":cache_store_mha_bf16_1to2_1k_smoke",
        ":cache_store_mha_bf16_2to1_1k_smoke",
        ":cache_store_mha_bf16_2to2_1k_smoke",
        ":cache_store_mha_int8_1to1_1k_smoke",
    ],
    tags = ["manual", "cache_smoke"],
)

sh_test(
    name = "cache_store_mha_bf16_invalid_port_recovery_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_bf16_invalid_port_recovery_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_bf16_invalid_port_recovery_1k.json"],
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "cache_smoke", "normal_cache_store", "fault_recovery"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mha_bf16_wrong_length_recovery_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_bf16_wrong_length_recovery_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_bf16_wrong_length_recovery_1k.json"],
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "cache_smoke", "normal_cache_store", "fault_recovery"],
    timeout = "long",
)

test_suite(
    name = "cache_store_reachable_fault_smoke",
    tests = [
        ":cache_store_mha_bf16_invalid_port_recovery_1k_smoke",
        ":cache_store_mha_bf16_wrong_length_recovery_1k_smoke",
    ],
    tags = ["manual", "cache_smoke"],
)

sh_test(
    name = "cache_store_mha_bf16_2to2_lifecycle_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_bf16_2to2_lifecycle_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_bf16_2to2_lifecycle_1k.json"],
    env = {"GPU_COUNT": "4"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

test_suite(
    name = "cache_store_reachable_lifecycle_smoke",
    tests = [":cache_store_mha_bf16_2to2_lifecycle_1k_smoke"],
    tags = ["manual", "cache_smoke"],
)

sh_test(
    name = "cache_store_mla_bf16_1to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mla_bf16_1to1_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mla_bf16_1to1_1k.json"],
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mla_fp8_1to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mla_fp8_1to1_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mla_fp8_1to1_1k.json"],
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

test_suite(
    name = "cache_store_reachable_mla_smoke",
    tests = [
        ":cache_store_mla_bf16_1to1_1k_smoke",
        ":cache_store_mla_fp8_1to1_1k_smoke",
    ],
    tags = ["manual", "cache_smoke"],
)

sh_test(
    name = "cache_store_mha_int8_1to2_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_int8_1to2_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_int8_1to2_1k.json"],
    env = {"GPU_COUNT": "3"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mha_int8_2to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_int8_2to1_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_int8_2to1_1k.json"],
    env = {"GPU_COUNT": "3"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mha_int8_2to2_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mha_int8_2to2_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mha_int8_2to2_1k.json"],
    env = {"GPU_COUNT": "4"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

test_suite(
    name = "cache_store_reachable_extended_smoke",
    tests = [
        ":cache_store_mha_int8_1to2_1k_smoke",
        ":cache_store_mha_int8_2to1_1k_smoke",
        ":cache_store_mha_int8_2to2_1k_smoke",
    ],
    tags = ["manual", "cache_smoke"],
)

sh_test(
    name = "cache_store_mla_bf16_cp2to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mla_bf16_cp2to1_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mla_bf16_cp2to1_1k.json"],
    env = {"GPU_COUNT": "3"},
    tags = ["manual", "cache_smoke", "normal_cache_store", "cp"],
    timeout = "long",
)

sh_test(
    name = "cache_store_mla_fp8_cp2to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_smoke_rank)", "$(location :configs/cache_store_mla_fp8_cp2to1_1k.json)"],
    data = [":cache_store_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_mla_fp8_cp2to1_1k.json"],
    env = {"GPU_COUNT": "3"},
    tags = ["manual", "cache_smoke", "normal_cache_store", "cp"],
    timeout = "long",
)

test_suite(
    name = "cache_store_reachable_cp_smoke",
    tests = [
        ":cache_store_mla_bf16_cp2to1_1k_smoke",
        ":cache_store_mla_fp8_cp2to1_1k_smoke",
    ],
    tags = ["manual", "cache_smoke"],
)

sh_test(
    name = "cache_store_next_bf16_1to1_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_next_smoke_rank)", "$(location :configs/cache_store_next_bf16_1to1_1k.json)"],
    data = [":cache_store_next_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_next_bf16_1to1_1k.json"],
    env = {"GPU_COUNT": "2"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

sh_test(
    name = "cache_store_next_bf16_2to2_1k_smoke",
    srcs = ["cache_store_smoke_entry.sh"],
    args = ["$(location :cache_store_next_smoke_rank)", "$(location :configs/cache_store_next_bf16_2to2_1k.json)"],
    data = [":cache_store_next_smoke_rank", "run_cache_store_smoke.py", "configs/cache_store_next_bf16_2to2_1k.json"],
    env = {"GPU_COUNT": "4"},
    tags = ["manual", "cache_smoke", "normal_cache_store"],
    timeout = "long",
)

cc_binary(
    name = "cache_store_next_smoke_rank",
    testonly = True,
    srcs = ["CacheStoreNextSmokeRank.cc", "CacheStoreSmokeEndpoint.h", "CacheStoreNextOracle.h", "CacheSmokeSupport.h"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/cache/connector/p2p:layer_cache_buffer",
        "//rtp_llm/cpp/disaggregate/cache_store:cache_store",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
)

test_suite(
    name = "cache_store_reachable_next_smoke",
    tests = [":cache_store_next_bf16_1to1_1k_smoke", ":cache_store_next_bf16_2to2_1k_smoke"],
    tags = ["manual", "cache_smoke"],
)
