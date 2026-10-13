load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

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

cc_test(
    name = "cuda_speculative_sampling_test",
    srcs = ["CudaSpeculativeSamplingTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/common/kernels:vocab_prune_kernels",
        "//rtp_llm/models_py/bindings/cuda/kernels:kernels_speculative_sampling",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

cc_test(
    name = "crc_block_copy_batch_test",
    srcs = ["CrcBlockCopyBatchTest.cc"],
    copts = copts() + select({
        "//:using_cuda13_x86": ["-DRTP_LLM_CRC_INTERNAL_TESTS=1"],
        "//:using_cuda13_arm": ["-DRTP_LLM_CRC_INTERNAL_TESTS=1"],
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache:crc_dump_writer",
        "//rtp_llm/models_py/bindings:crc_block_copy",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    exec_properties = select({
        "//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13"},
        "//conditions:default": {"gpu": "H20_CU13"},
    }),
    tags = ["exclusive-if-local"],
)

# Opt-in performance/correctness reproduction, separate from the regular CRC UT.
CUDA13_ONLY = select({
    "//:using_cuda13_x86": [],
    "//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

cc_library(
    name = "main_staged_copy_kernel",
    testonly = True,
    srcs = ["MainStagedCopyKernel.cu"],
    hdrs = ["MainStagedCopyKernel.h"],
    copts = cuda_copts(),
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "crc_copy_benchmark_support",
    testonly = True,
    srcs = [
        "CrcCopyBenchmarkSupport.cc",
        "MainStagedCopyBaseline.cc",
    ],
    hdrs = [
        "CrcCopyBenchmarkSupport.h",
        "MainStagedCopyBaseline.h",
    ],
    copts = copts(),
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [
        ":main_staged_copy_kernel",
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/cache:device_block_pool_config_helper",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/models_py/bindings:crc_block_copy_hdr",
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/cuda:no_block_copy",
        "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
)

cc_binary(
    name = "crc_copy_benchmark",
    testonly = True,
    srcs = ["CrcCopyBenchmark.cu"],
    copts = cuda_copts() + ["-O3"],
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [
        ":crc_copy_benchmark_support",
        "//rtp_llm/models_py/bindings:crc_block_copy",
        "//rtp_llm/models_py/bindings/common/kernels:copy_tile_kernel",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

# Test-only source generation changes the runtime segmentation decision while
# retaining production workspace capacity, copy calls, kernels and checks.
py_binary(
    name = "generate_crc_segmentation_backend",
    testonly = True,
    srcs = ["generate_crc_segmentation_backend.py"],
    tags = ["manual"],
)

genrule(
    name = "crc_segmentation_backend_source",
    testonly = True,
    srcs = [
        "//rtp_llm/models_py/bindings:CrcBlockCopy.h",
        "//rtp_llm/models_py/bindings/cuda:CrcBlockCopy.cu",
        "//rtp_llm/models_py/bindings/cuda:CrcBlockCopyInternal.cuh",
    ],
    outs = [
        "ForcedCrcBlockCopy.h",
        "ForcedCrcBlockCopy.cu",
        "ForcedCrcBlockCopyInternal.cuh",
        "crc_segmentation_backend_manifest.json",
    ],
    cmd = "$(location :generate_crc_segmentation_backend) " +
          "--header $(location //rtp_llm/models_py/bindings:CrcBlockCopy.h) " +
          "--source $(location //rtp_llm/models_py/bindings/cuda:CrcBlockCopy.cu) " +
          "--internal $(location //rtp_llm/models_py/bindings/cuda:CrcBlockCopyInternal.cuh) " +
          "--output-header $(location :ForcedCrcBlockCopy.h) " +
          "--output-source $(location :ForcedCrcBlockCopy.cu) " +
          "--output-internal $(location :ForcedCrcBlockCopyInternal.cuh) " +
          "--manifest $(location :crc_segmentation_backend_manifest.json)",
    tools = [":generate_crc_segmentation_backend"],
    tags = ["manual"],
)

cc_library(
    name = "forced_crc_block_copy",
    testonly = True,
    srcs = ["ForcedCrcBlockCopy.cu"],
    hdrs = ["ForcedCrcBlockCopy.h", "ForcedCrcBlockCopyInternal.cuh"],
    copts = cuda_copts(),
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache:crc_dump_writer",
        "//rtp_llm/models_py/bindings:crc_block_copy_hdr",
        "//rtp_llm/models_py/bindings/common/kernels:copy_tile_kernel",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + select({
        "//:using_cuda13_arm": ["@nvcomp_cuda13_arm//:crc32"],
        "//conditions:default": ["@nvcomp_cuda13_x86//:crc32"],
    }),
)

cc_library(
    name = "crc_segmentation_benchmark_harness",
    testonly = True,
    textual_hdrs = ["CrcCopyBenchmark.cu"],
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [
        ":crc_copy_benchmark_support",
        "//rtp_llm/models_py/bindings:crc_block_copy",
        "//rtp_llm/models_py/bindings/common/kernels:copy_tile_kernel",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "crc_segmentation_benchmark",
    testonly = True,
    srcs = ["CrcSegmentationBenchmark.cu"],
    copts = cuda_copts() + ["-O3"],
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [":crc_segmentation_benchmark_harness", ":forced_crc_block_copy"],
)

py_test(
    name = "crc_segmentation_benchmark_test",
    srcs = ["crc_segmentation_benchmark_stats.py", "run_crc_segmentation_benchmark.py"],
    main = "run_crc_segmentation_benchmark.py",
    data = [
        ":crc_segmentation_benchmark",
        ":ForcedCrcBlockCopy.h",
        ":ForcedCrcBlockCopy.cu",
        ":ForcedCrcBlockCopyInternal.cuh",
        ":crc_segmentation_backend_manifest.json",
    ],
    args = [
        "--binary=$(location :crc_segmentation_benchmark)",
        "--backend-artifact=$(location :ForcedCrcBlockCopy.h)",
        "--backend-artifact=$(location :ForcedCrcBlockCopy.cu)",
        "--backend-artifact=$(location :ForcedCrcBlockCopyInternal.cuh)",
        "--backend-artifact=$(location :crc_segmentation_backend_manifest.json)",
    ],
    size = "large",
    timeout = "long",
    tags = ["manual", "gpu", "local", "exclusive", "no-remote"],
    target_compatible_with = CUDA13_ONLY,
)

py_test(
    name = "crc_segmentation_benchmark_stats_test",
    srcs = ["crc_segmentation_benchmark_stats.py", "crc_segmentation_benchmark_stats_test.py"],
    main = "crc_segmentation_benchmark_stats_test.py",
)

# Diagnostic-only linker interception; the ordinary benchmark has no wrapper.
cc_binary(
    name = "staged_memcpy_profile",
    testonly = True,
    srcs = ["CrcCopyBenchmark.cu"],
    copts = cuda_copts() + ["-O3", "-DRTP_LLM_STAGED_MEMCPY_PROFILE=1"],
    linkopts = [
        "-Wl,--wrap=memcpy",
        "-Wl,--defsym=__wrap_memcpy=stagedProfileMemcpy",
    ],
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [
        ":crc_copy_benchmark_support",
        ":staged_memcpy_profiler",
        "//rtp_llm/models_py/bindings:crc_block_copy",
        "//rtp_llm/models_py/bindings/common/kernels:copy_tile_kernel",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "staged_memcpy_profiler",
    testonly = True,
    srcs = ["StagedMemcpyProfiler.cc"],
    hdrs = ["StagedMemcpyProfiler.h"],
    copts = copts(),
    tags = ["manual"],
)

py_test(
    name = "staged_memcpy_profile_test",
    srcs = [
        "crc_copy_benchmark_stats.py",
        "run_crc_copy_benchmark.py",
        "staged_memcpy_profile_stats.py",
    ],
    main = "run_crc_copy_benchmark.py",
    data = [":staged_memcpy_profile"],
    args = ["--binary=$(location :staged_memcpy_profile)", "--staged-memcpy-profile"],
    size = "large",
    timeout = "long",
    tags = ["manual", "gpu", "local", "exclusive", "no-remote"],
    target_compatible_with = CUDA13_ONLY,
)

py_test(
    name = "staged_memcpy_profile_stats_test",
    srcs = [
        "crc_copy_benchmark_stats.py",
        "crc_copy_benchmark_stats_test.py",
        "staged_memcpy_profile_stats.py",
        "staged_memcpy_profile_stats_test.py",
    ],
    main = "staged_memcpy_profile_stats_test.py",
)

py_test(
    name = "crc_copy_benchmark_test",
    srcs = [
        "crc_copy_benchmark_stats.py",
        "run_crc_copy_benchmark.py",
    ],
    main = "run_crc_copy_benchmark.py",
    data = [":crc_copy_benchmark"],
    args = ["--binary=$(location :crc_copy_benchmark)"],
    size = "large",
    timeout = "long",
    tags = ["manual", "gpu", "local", "exclusive", "no-remote"],
    target_compatible_with = CUDA13_ONLY,
)

py_test(
    name = "crc_copy_benchmark_stats_test",
    srcs = [
        "crc_copy_benchmark_stats.py",
        "crc_copy_benchmark_stats_test.py",
    ],
    main = "crc_copy_benchmark_stats_test.py",
)

cc_test(
    name = "copy_tile_kernel_test",
    srcs = ["CopyTileKernelTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings/common/kernels:copy_tile_kernel",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    exec_properties = select({
        "//:using_cuda13_arm": {"gpu": "SM100_ARM_CU13"},
        "//conditions:default": {"gpu": "H20_CU13"},
    }),
    tags = ["exclusive-if-local"],
)

# Reproduce the shared kernel's cold-cache thread sweep. Raw paired samples and
# summary.csv are preserved in Bazel's test.outputs/outputs.zip.
py_test(
    name = "tile_copy_thread_profile_test",
    srcs = [
        "crc_copy_benchmark_stats.py",
        "run_crc_copy_benchmark.py",
        "run_tile_copy_thread_profile.py",
    ],
    main = "run_tile_copy_thread_profile.py",
    data = [":crc_copy_benchmark"],
    args = ["--binary=$(location :crc_copy_benchmark)"],
    size = "large",
    timeout = "long",
    tags = ["manual", "gpu", "local", "exclusive", "no-remote"],
    target_compatible_with = CUDA13_ONLY,
)

py_test(
    name = "tile_copy_thread_profile_stats_test",
    srcs = [
        "crc_copy_benchmark_stats.py",
        "run_crc_copy_benchmark.py",
        "run_tile_copy_thread_profile.py",
        "tile_copy_thread_profile_stats_test.py",
    ],
    main = "tile_copy_thread_profile_stats_test.py",
)
