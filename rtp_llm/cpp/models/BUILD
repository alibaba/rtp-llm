load("//:def.bzl", "copts")

# Sample infos - model sampling data structures
cc_library(
    name = "sample_infos",
    hdrs = [
        "SampleInfos.h",
    ],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/utils:tensor_debug_utils",
    ],
    visibility = ["//visibility:public"],
)

# Position IDs generator - model inference component
cc_library(
    name = "position_ids_generator",
    hdrs = [
        "position_ids/PositionIdsGenerator.h",
    ],
    srcs = [
        "position_ids/PositionIdsGenerator.cc",
    ],
    deps = [
        "//:rtp_compute_ops"
    ],
    visibility = ["//visibility:public"],
)

# DFA utilities - includes PrefixToCandidateTokens functionality
cc_library(
    name = "dfa_utils",
    hdrs = [
        "logits_processor/DFAUtil.h",
        "logits_processor/PrefixToCandidateTokens.h",
    ],
    deps = [
        "//:rtp_compute_ops",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

# Logits processor - model output processing
cc_library(
    name = "logits_processor",
    hdrs = [
        "logits_processor/BaseLogitsProcessor.h",
        "logits_processor/MultiSeqLogitsProcessor.h",
        "logits_processor/LogitsProcessorStates.h",
        "logits_processor/ThinkModeLogitsProcessor.h",
        "logits_processor/TreeLogitsProcessor.h",
        "logits_processor/LogitsProcessorFactory.h",
        "logits_processor/xgrammar/XGrammarCompilerCache.h",
        "logits_processor/xgrammar/XGrammarGpuState.h",
        "logits_processor/xgrammar/XGrammarLogitsProcessor.h",
    ] + select({
        "//:using_cuda13_x86": [
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/dlpack/dlpack.h",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/compiler.h",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/config.h",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/exception.h",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/grammar.h",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/matcher.h",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/object.h",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/tokenizer_info.h",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/xgrammar.h",
        ],
        "//:using_cuda13_arm": [
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/dlpack/dlpack.h",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/compiler.h",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/config.h",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/exception.h",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/grammar.h",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/matcher.h",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/object.h",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/tokenizer_info.h",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/include/xgrammar/xgrammar.h",
        ],
        "//conditions:default": [],
    }),
    srcs = [
        "logits_processor/BaseLogitsProcessor.cc",
        "logits_processor/MultiSeqLogitsProcessor.cc",
        "logits_processor/LogitsProcessorStates.cc",
        "logits_processor/ThinkModeLogitsProcessor.cc",
        "logits_processor/TreeLogitsProcessor.cc",
        "logits_processor/LogitsProcessorFactory.cc",
        "logits_processor/xgrammar/XGrammarCompilerCache.cc",
        "logits_processor/xgrammar/XGrammarLogitsProcessor.cc",
    ],
    deps = [
        ":sample_infos",
        ":dfa_utils",
        "//rtp_llm/cpp/engine_base/stream:generate_types",
        "//rtp_llm/cpp/engine_base/stream:generate_config",
        "//rtp_llm/cpp/multimodal_processor:multimodal_types",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/utils:error_code",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:standalone_ops",
            "//rtp_llm/models_py/bindings/cuda/kernels:xgrammar_kernels",
        ],
        "//conditions:default": [],
    }),
    additional_linker_inputs = select({
        "//:using_cuda13_x86": [
            "@pip_gpu_cuda13_torch_apache_tvm_ffi//:site-packages/tvm_ffi/lib/libtvm_ffi.so",
            "@pip_gpu_cuda13_torch_xgrammar//:site-packages/xgrammar/libxgrammar_bindings.so",
        ],
        "//:using_cuda13_arm": [
            "@pip_cuda13_arm_torch_apache_tvm_ffi//:site-packages/tvm_ffi/lib/libtvm_ffi.so",
            "@pip_cuda13_arm_torch_xgrammar//:site-packages/xgrammar/libxgrammar_bindings.so",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//:using_cuda13_x86": [
            "external/pip_gpu_cuda13_torch_xgrammar/site-packages/xgrammar/libxgrammar_bindings.so",
            "external/pip_gpu_cuda13_torch_apache_tvm_ffi/site-packages/tvm_ffi/lib/libtvm_ffi.so",
            "-Wl,-rpath,$$ORIGIN/../../../../../..//external/pip_gpu_cuda13_torch_xgrammar/site-packages/xgrammar",
            "-Wl,-rpath,$$ORIGIN/../../../../../..//external/pip_gpu_cuda13_torch_apache_tvm_ffi/site-packages/tvm_ffi/lib",
        ],
        "//:using_cuda13_arm": [
            "external/pip_cuda13_arm_torch_xgrammar/site-packages/xgrammar/libxgrammar_bindings.so",
            "external/pip_cuda13_arm_torch_apache_tvm_ffi/site-packages/tvm_ffi/lib/libtvm_ffi.so",
            "-Wl,-rpath,$$ORIGIN/../../../../../..//external/pip_cuda13_arm_torch_xgrammar/site-packages/xgrammar",
            "-Wl,-rpath,$$ORIGIN/../../../../../..//external/pip_cuda13_arm_torch_apache_tvm_ffi/site-packages/tvm_ffi/lib",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "//:using_cuda13_x86": [
            "-DRTP_LLM_ENABLE_XGRAMMAR_CPP=1",
            "-isystem",
            "external/pip_gpu_cuda13_torch_xgrammar/site-packages/xgrammar/include",
        ],
        "//:using_cuda13_arm": [
            "-DRTP_LLM_ENABLE_XGRAMMAR_CPP=1",
            "-isystem",
            "external/pip_cuda13_arm_torch_xgrammar/site-packages/xgrammar/include",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# Context Parallel Interface - base interface for context parallel processors
cc_library(
    name = "context_parallel_interface",
    hdrs = [
        "context_parallel/ContextParallelProcessorBase.h",
    ],
    deps = [
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "context_parallel",
    hdrs = [
        "context_parallel/ZigzagProcessor.h",
    ],
    srcs = [
        "context_parallel/ContextParallelProcessorBase.cc",
        "context_parallel/ZigzagProcessor.cc",
        "context_parallel/ContextParallelProcessorFactory.cc",
    ],
    deps = [
        ":context_parallel_interface",
        "//:rtp_compute_ops",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "models",
    hdrs = glob([
        "*.h",
    ]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        ":sample_infos",
        ":logits_processor",
        ":dfa_utils",
        ":context_parallel",
        "//:rtp_compute_ops",
        "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
        "//rtp_llm/cpp/engine_base/stream:complete_token_ids",
        "//rtp_llm/cpp/cache:cache_types",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/models_py/bindings/core:cache_store_async_writer",
        "//rtp_llm/cpp/utils:debug_utils",
        "//rtp_llm/cpp/utils:profiling_scope",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_impl",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/cuda_graph:cuda_graph_hdrs_lib",
        "//rtp_llm/models_py/bindings/common:fuse_copy_op",
    ] + select({
        "//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/kernels:attention_input_metadata",
            "//rtp_llm/models_py/bindings/cuda:bf16_gemm_op"
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# Expert statistics - monitoring component
cc_library(
    name = "stats",
    hdrs = [
        "eplb/stats/ExpertStats.h",
    ],
    visibility = ["//visibility:public"],
)

# EPLB configuration
cc_library(
    name = "eplb_config",
    hdrs = [
        "eplb/EplbConfig.h",
    ],
    visibility = ["//visibility:public"],
)

# Expert Load Balancer
cc_library(
    name = "eplb",
    hdrs = [
        "eplb/ExpertBalancer.h",
        "eplb/ExpertBalancerPythonWrapper.h",
    ],
    srcs = [
        "eplb/ExpertBalancer.cc",
        "eplb/ExpertBalancerPythonWrapper.cc",
    ],
    deps = [
        ":models",
        ":dfa_utils",
        "//rtp_llm/cpp/config:eplb_config",
        "//rtp_llm/cpp/metrics",
        "//rtp_llm/cpp/pybind:py_utils",
        "//rtp_llm/cpp/cache:cache_types",
    ],
    visibility = ["//visibility:public"],
)
