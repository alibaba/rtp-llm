load("//:def.bzl", "copts")

CUDA_ONLY = select({
    "@//:using_cuda": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

cc_library(
    name = "transfer_benchmark_workload",
    srcs = ["TransferBenchmarkWorkload.cc"],
    hdrs = ["TransferBenchmarkWorkload.h"],
    copts = copts(),
    tags = ["manual"],
)

cc_library(
    name = "tree_workload_generator",
    srcs = ["TreeWorkloadGenerator.cc"],
    hdrs = ["TreeWorkloadGenerator.h"],
    copts = copts(),
    tags = ["manual"],
)

cc_library(
    name = "online_tree_scheduler",
    srcs = ["OnlineTreeScheduler.cc"],
    hdrs = ["OnlineTreeScheduler.h"],
    copts = copts(),
    tags = ["manual"],
    deps = [
        ":tree_workload_generator",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "benchmark_common",
    srcs = [
        "BenchmarkJsonWriter.cc",
        "CommonBenchmarkOptions.cc",
        "ModelProfile.cc",
        "TreeBenchmarkOptions.cc",
        "TransferBenchmarkOptions.cc",
    ],
    hdrs = [
        "BenchmarkArgumentParser.h",
        "BenchmarkJsonWriter.h",
        "CommonBenchmarkOptions.h",
        "ModelProfile.h",
        "TreeBenchmarkOptions.h",
        "TransferBenchmarkOptions.h",
    ],
    copts = copts(),
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [
        "@boringssl//:crypto",
        "@rapidjson//:rapidjson",
    ],
)

cc_library(
    name = "benchmark_fixture",
    target_compatible_with = CUDA_ONLY,
    srcs = [
        "BenchmarkFixture.cc",
    ],
    hdrs = [
        "BenchmarkFixture.h",
    ],
    copts = copts(),
    tags = ["manual"],
    deps = [
        ":benchmark_common",
        ":tree_workload_generator",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache:device_block_pool",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/cache:kv_cache_specs",
        "//rtp_llm/cpp/cache:kv_cache_allocator_hdr",
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/utils:core_utils",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tree_benchmark_runner",
    target_compatible_with = CUDA_ONLY,
    srcs = [
        "TreeBenchmarkRunner.cc",
    ],
    hdrs = [
        "TreeBenchmarkRunner.h",
    ],
    copts = copts() + ["-fno-access-control"],
    tags = ["manual"],
    deps = [
        ":benchmark_common",
        ":benchmark_fixture",
        ":online_tree_scheduler",
        ":tree_workload_generator",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)

cc_library(
    name = "transfer_benchmark_runner",
    target_compatible_with = CUDA_ONLY,
    srcs = [
        "TransferBenchmarkRunner.cc",
    ],
    hdrs = [
        "TransferBenchmarkRunner.h",
    ],
    copts = copts() + ["-fno-access-control"],
    tags = ["manual"],
    deps = [
        ":benchmark_common",
        ":benchmark_fixture",
        ":transfer_benchmark_workload",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache:device_block_pool",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:cache_async_context",
        "//rtp_llm/cpp/utils:core_utils",
    ] + select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "transfer_benchmark_workload_test",
    srcs = ["TransferBenchmarkWorkloadTest.cc"],
    copts = copts(),
    deps = [
        ":transfer_benchmark_workload",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "online_tree_scheduler_test",
    target_compatible_with = CUDA_ONLY,
    exec_properties = {"gpu": "A10"},
    srcs = ["OnlineTreeSchedulerTest.cc"],
    copts = copts() + ["-fno-access-control"],
    deps = [
        ":online_tree_scheduler",
        ":benchmark_fixture",
        ":tree_benchmark_runner",
        ":tree_workload_generator",
        "//rtp_llm/cpp/cache/block_tree_cache/test:block_tree_cache_test_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "@//:using_cuda": ["//3rdparty:cuda_driver"],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "block_tree_cache_gpu_benchmark",
    target_compatible_with = CUDA_ONLY,
    srcs = [
        "BlockTreeCacheBenchmarkMain.cc",
    ],
    copts = copts(),
    tags = ["manual", "local", "no-remote"],
    exec_properties = {"gpu": "H20"},
    deps = [
        ":benchmark_cli",
        ":transfer_benchmark_runner",
        ":tree_benchmark_runner",
    ] + select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "//conditions:default": [],
    }),
)

py_library(
    name = "benchmark_driver_sources",
    srcs = [
        "benchmark_cases.py",
        "run_block_tree_cache_benchmark.py",
    ],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "benchmark_driver_profile_test",
    srcs = [
        "benchmark_cases.py",
        "run_block_tree_cache_benchmark.py",
        "run_block_tree_cache_benchmark_test.py",
    ],
    main = "run_block_tree_cache_benchmark_test.py",
    data = [
        "docs/benchmark_cases.md",
        "docs/nsys_collection_guide.md",
        "//rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles:deepseek_v4_pro_fp8_tp1_cp1",
        "//rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles:deepseek_v4_flash_fp8_tp1_cp1_tpb1024",
    ],
)

py_binary(
    name = "block_tree_cache_benchmark_driver",
    target_compatible_with = CUDA_ONLY,
    srcs = [
        "run_block_tree_cache_benchmark.py",
        "benchmark_cases.py",
    ],
    main = "run_block_tree_cache_benchmark.py",
    tags = ["manual"],
    data = [
        ":block_tree_cache_gpu_benchmark",
        "//rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles:deepseek_v4_flash_fp8_tp1_cp1_tpb1024",
    ],
    deps = [],
)

py_test(
    name = "block_tree_cache_benchmark_smoke_test",
    target_compatible_with = CUDA_ONLY,
    srcs = ["run_block_tree_cache_benchmark_smoke_test.py"],
    main = "run_block_tree_cache_benchmark_smoke_test.py",
    data = [
        ":block_tree_cache_gpu_benchmark",
        "benchmark_cases.py",
        "run_block_tree_cache_benchmark.py",
        "//rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles:deepseek_v4_flash_fp8_tp1_cp1_tpb1024",
    ],
    # Runs the CUDA benchmark binary; keep on GPU machines and off remote exec.
    tags = ["gpu", "local", "exclusive", "manual"],
    timeout = "moderate",
)

cc_test(
    name = "tree_workload_generator_test",
    srcs = ["TreeWorkloadGeneratorTest.cc"],
    copts = copts(),
    deps = [
        ":tree_workload_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "benchmark_cli",
    srcs = ["BenchmarkCli.cc"],
    hdrs = ["BenchmarkCli.h"],
    copts = copts(),
    deps = [":benchmark_common"],
)

cc_test(
    name = "benchmark_cli_test",
    srcs = ["BenchmarkCliTest.cc"],
    copts = copts(),
    data = ["//rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles:deepseek_v4_pro_fp8_tp1_cp1"],
    deps = [":benchmark_cli", "@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "model_profile_test",
    srcs = ["ModelProfileTest.cc"],
    copts = copts(),
    data = [
        "//rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles:deepseek_v4_pro_fp8_tp1_cp1",
        "//rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles:deepseek_v4_flash_fp8_tp1_cp1_tpb1024",
    ],
    deps = [":benchmark_common", "@com_google_googletest//:gtest_main"],
)
