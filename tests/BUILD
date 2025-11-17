load("//:def.bzl", "copts", "cuda_copts", "rocm_copts")
load("//bazel:arch_select.bzl", "torch_deps", "fa_deps", "cuda_register")

################################ py test ################################
fa_deps()
cuda_register()

cc_library(
    name = "test_ops_libs",
    srcs = glob([
        "layernorm/*.cpp",
        "rotary_embedding/*.cpp",
        "attention_logn/*.cpp",
        "gemm_group/*.cc",
        "gemm/*.cc",
        "mla/*.cc",
        "unittests/test_activation.cu",
    ]) + select({
        "@//:using_cuda12": glob([
            "fp8_gemm/*.cc",
            "gemm_dequantize/*.cc",
            "int8_gemm/*.cc",
        ]),
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm/cpp/cuda:memory_utils",
        "//rtp_llm/cpp/cuda:allocator_torch",
        "//rtp_llm/cpp/pybind:th_compute_lib",
    ] + torch_deps(),
    copts = cuda_copts(),
    alwayslink = True,
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_ops",
    deps = select({
        "@//:using_cuda12":[":test_ops_libs"],
        "@//:using_rocm": [":rocm_test_ops_libs"],
        "//conditions:default": [],
    }),
    data = [
        "//:rtp_compute_ops",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

op_test_data = [
    "//:th_transformer",
    "//:rtp_compute_ops",
    ":test_ops",
    "//rtp_llm:rtp_llm_lib",
]

op_test_deps = [
    "//rtp_llm:torch",
    "//rtp_llm:numpy",
    "//rtp_llm:transformers",
]

py_test(
    name = "generalT5LayerNorm",
    srcs = [
        "layernorm/generalT5LayerNorm.py"
    ],
    data = op_test_data,
    deps = op_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "generalLayerNorm",
    srcs = [
        "layernorm/generalLayerNorm.py"
    ],
    data = op_test_data,
    deps = op_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "rotary_position_embedding",
    srcs = [
        "rotary_embedding/rotary_position_embedding.py",
        "rotary_embedding/yarn_rotary_embedding.py",
        "rotary_embedding/deepseek_yarn_rotary_embedding.py",
    ],
    data = op_test_data,
    deps = op_test_deps + [
        "//rtp_llm:einops"
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "logn_attention",
    srcs = [
        "attention_logn/logn_attention.py"
    ],
    data = op_test_data,
    deps = op_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "th_int8_gemm",
    srcs = [
        "int8_gemm/th_int8_gemm.py"
    ],
    data = op_test_data,
    deps = op_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "gemm_op_test",
    srcs = [
        "gemm/gemm_op_test.py"
    ],
    data = op_test_data,
    deps = op_test_deps,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "th_fp8_gemm",
    srcs = [
        "fp8_gemm/th_fp8_gemm.py"
    ],
    data = op_test_data,
    deps = op_test_deps,
    tags = ["H20"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "th_gemm_dequantize",
    srcs = [
        "gemm_dequantize/th_gemm_dequantize.py"
    ],
    data = op_test_data,
    deps = op_test_deps + [
        "//rtp_llm:device",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "merge_transpose_test",
    srcs = [
        "mla/merge_transpose_test.py"
    ],
    data = op_test_data,
    deps = op_test_deps,
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "mla_context_attention",
    srcs = [
        "mla/mla_context_attention.py",
        "mla/test_util.py"
    ],
    data = op_test_data,
    deps = op_test_deps + [
        "//rtp_llm:flash_attn",
    ],
    exec_properties = {'gpu':'A10'},
)



py_test(
    name = "mla_decode_attention",
    srcs = [
        "mla/mla_decode_attention.py",
        "mla/test_util.py"
    ],
    data = op_test_data,
    deps = op_test_deps + [
        "//rtp_llm:rtp_llm_lib",
    ],
    exec_properties = {'gpu':'A10'},
)

cc_library(
    name = "rocm_test_ops_libs",
    srcs = glob([
        "mla/mla_rotary_kvcache_test.cc",
        "mla/mla_decoder_attention.cc",
        "mla/mla_context_attention.cc",
        "gemm/gemm_op_test.cc",
        "ffn/rocm_ffn_moe_fp8_test.cc",
        "ffn/rocm_ffn_moe_fp8_ptpc_test.cc",
        "ffn/rocm_ffn_moe_bf16_test.cc",
        "layernorm/fusedQkRmsNorm.cpp",
        "deepep/rocm_ffn_moe_fp8_ll_test.cc",
    ]),
    deps = torch_deps() + [
        "//rtp_llm/cpp/devices/rocm_impl:rocm_impl",
    ],
    copts = rocm_copts() + [
    "-DUSE_ROCM=1",
    "-DDISABLE_INTERNODE=1",
    "-DFORCE_NVSHMEM_API=1"],
    alwayslink = True,
    visibility = ["//visibility:public"],
	  tags = ["rocm"],
)

cc_binary(
    name = "rocm_test_ops",
    deps = [":rocm_test_ops_libs"],
    linkshared = 1,
    visibility = ["//visibility:public"],
    tags = ["rocm"],
)

py_test(
    name = "rocm_gemm_op_test",
    srcs = [
        "gemm/rocm_gemm_op_test.py"
    ],
    data = [
        "//:th_transformer",
        ":rocm_test_ops",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:rtp_llm_lib",
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "128000000",
    },
    tags = [
        "rocm",
    ],
    timeout = "eternal",
)

py_test(
    name = "rocm_ptpc_a8w8_gemm_op_test",
    srcs = [
        "gemm/rocm_ptpc_a8w8_gemm_op_test.py"
    ],
    data = [
        "//:th_transformer",
        ":rocm_test_ops",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:rtp_llm_lib",
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
    },
    tags = [
        "rocm",
    ],
)

py_test(
    name = "rocm_ptpc_gemm_op_test",
    srcs = [
        "gemm/rocm_ptpc_gemm_op_test.py"
    ],
    data = [
        "//:th_transformer",
        ":rocm_test_ops",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:rtp_llm_lib",
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
    },
    tags = [
        "rocm",
    ],
)

py_test(
    name = "rocm_mla_decode_attention",
    srcs = [
        "mla/rocm_mla_decode_attention.py",
        "mla/test_util.py"
    ],
    data = [
        "//:th_transformer",
        ":rocm_test_ops",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:rtp_llm_lib",
    ],
    tags = [
        "rocm",
    ],
)

#ToDo: fix aiter import issue
#py_test(
#    name = "rocm_ffn_moe_fp8_test",
#    srcs = [
#        "ffn/rocm_ffn_moe_fp8_test.py"
#    ],
#    data = [
#        "//:th_transformer",
#        ":rocm_test_ops",
#    ],
#    deps = [
#        "//rtp_llm:torch",
#        "//rtp_llm:rtp_llm_lib",
#        "@aiter//:aiter_py",
#    ],
#    env = {
#        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
#    },
#    tags = [
#        "rocm",
#    ],
#)

py_test(
   name = "rocm_ffn_moe_fp8_ll_test",
   srcs = [
       "deepep/rocm_ffn_moe_fp8_ll_test.py"
   ],
   data = [
       "//:th_transformer",
       ":rocm_test_ops",
       "//rtp_llm/cpp/rocm:rocm_custom_ar"
   ],
   deps = [
       "//rtp_llm:torch",
       "//rtp_llm:rtp_llm_lib",
    #    "@aiter//:aiter_py",
   ],
   env = {
       "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
       "USE_DEEPEP_MOE": "1",
       "USE_DEEPEP_LOW_LATENCY": "1",
   },
   tags = ["rocm"],
)

#ToDo: fix aiter import issue
# py_test(
#     name = "rocm_ffn_moe_fp8_ptpc_test",
#     srcs = [
#         "ffn/rocm_ffn_moe_fp8_ptpc_test.py"
#     ],
#     data = [
#         "//:th_transformer",
#         ":rocm_test_ops",
#     ],
#     deps = [
#         "//rtp_llm:torch",
#         "//rtp_llm:rtp_llm_lib",
#     ],
#     env = {
#         "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
#     },
#     tags = [
#         "rocm",
#     ],
# )

#ToDo: fix aiter import issue
#py_test(
#    name = "rocm_ffn_moe_bf16_test",
#    srcs = [
#        "ffn/rocm_ffn_moe_bf16_test.py"
#    ],
#    data = [
#        "//:th_transformer",
#        ":rocm_test_ops",
#    ],
#    deps = [
#        "//rtp_llm:torch",
#        "//rtp_llm:rtp_llm_lib",
#        "@aiter//:aiter_py",
#    ],
#    env = {
#        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
#    },
#    tags = ["rocm"],
#)

py_test(
    name = "mla_rotary_kvcache_test",
    srcs = [
        "mla/mla_rotary_kvcache_test.py",
        "mla/test_util.py",
        "mla/rotary_util.py"
    ],
    data = op_test_data,
    deps = op_test_deps + [
        "//rtp_llm:rtp_llm_lib",
    ],
    exec_properties = {'gpu':'A10'},
)


py_test(
    name = "bench_mla_attention_layer",
    srcs = [
        "mla/bench_mla_attention_layer.py",
        "mla/test_util.py"
    ],
    data = op_test_data,
    deps = op_test_deps + [
        "//rtp_llm:rtp_llm_lib",
    ],
    tags = [
        "manual",
    ],
)

py_test(
    name = "rocm_mla_context_attention",
    srcs = [
        "mla/rocm_mla_context_attention.py"
    ],
    data = [
        "//:th_transformer",
        ":rocm_test_ops",
    ],
   deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:rtp_llm_lib",
    ],
    tags = [
        "manual",
		"rocm",
    ],
)

py_test(
    name = "rocm_mla_rotary_kvcache_test",
    srcs = [
        "mla/rocm_mla_rotary_kvcache_test.py",
        "mla/test_util.py",
        "mla/rotary_util.py"
    ],
    data = [
        "//:th_transformer",
        ":rocm_test_ops",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:rtp_llm_lib",
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
    },
    tags = [
        "rocm",
    ],
)

cc_test(
    name = "eplb_kernel_test",
    srcs = [
        "eplb/eplb_kernel_test.cc",
    ],
    copts = cuda_copts(),
    deps = [
        "@com_google_googletest//:gtest",
        "//rtp_llm/cpp/kernels:kernels_cu",
        "@local_config_cuda//cuda:cuda_driver",
    ] + torch_deps(),
    data = [
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "fused_qk_rmsnorm",
    srcs = [
        "layernorm/fused_qk_rmsnorm.py"
    ],
    data = op_test_data,
    deps = op_test_deps + [
        "//rtp_llm:rtp_llm_lib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "rocm_pertensor_int8_gemm_op_test",
    srcs = [
        "gemm/rocm_pertensor_int8_gemm_op_test.py"
    ],
    data = [
        "//:th_transformer",
        ":rocm_test_ops",
    ],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:rtp_llm_lib",
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
    },
    tags = [
        "rocm",
    ],
)