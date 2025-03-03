load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")


################################ py test ################################

cc_library(
    name = "test_ops_libs",
    srcs = glob([
        "layernorm/*.cpp",
        "rotary_embedding/*.cpp",
        "attention_logn/*.cpp",
        "gemm_dequantize/*.cc",
        "gemm_group/*.cc",
        "int8_gemm/*.cc",
        "mla/*.cc",
	    "fp8_gemm/*.cc",
        "unittests/test_activation.cu",
    ]),
    deps = torch_deps() + [
        "//src/fastertransformer/cuda:allocator_torch",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/kernels:kernels",
        "//:th_utils",
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//src/fastertransformer/devices/cuda_impl:cuda_impl",
        "@flash_attention//:flash_attention2_impl",
        "//:gpt_init_params",
    ],
    copts = cuda_copts(),
    alwayslink = True,
)

cc_binary(
    name = "test_ops",
    deps = [":test_ops_libs"],
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_test(
    name = "generalT5LayerNorm",
    srcs = [
        "layernorm/generalT5LayerNorm.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:transformers",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "generalLayerNorm",
    srcs = [
        "layernorm/generalLayerNorm.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "rotary_position_embedding",
    srcs = [
        "rotary_embedding/rotary_position_embedding.py",
        "rotary_embedding/yarn_rotary_embedding.py",
        "rotary_embedding/deepseek_yarn_rotary_embedding.py",
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:transformers",
        "//maga_transformer:einops"
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "logn_attention",
    srcs = [
        "attention_logn/logn_attention.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "th_int8_gemm",
    srcs = [
        "int8_gemm/th_int8_gemm.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops"
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:numpy",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "th_fp8_gemm",
    srcs = [
        "fp8_gemm/th_fp8_gemm.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:numpy",
    ],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "th_gemm_dequantize",
    srcs = [
        "gemm_dequantize/th_gemm_dequantize.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:numpy",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "merge_transpose_test",
    srcs = [
        "mla/merge_transpose_test.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "mla_gemm_test",
    srcs = [
        "mla/mla_gemm_test.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
    },
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "mla_context_attention",
    srcs = [
        "mla/mla_context_attention.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
    },
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "mla_decode_attention",
    srcs = [
        "mla/mla_decode_attention.py",
        "mla/test_util.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:maga_transformer_lib",
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "1024000",
    },
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "mla_rotary_kvcache_test",
    srcs = [
        "mla/mla_rotary_kvcache_test.py",
        "mla/test_util.py"
    ],
    data = [
        "//:th_transformer",
        ":test_ops",
    ],
    deps = [
        "//maga_transformer:torch",
        "//maga_transformer:maga_transformer_lib",
    ],
    exec_properties = {'gpu':'A10'},
)