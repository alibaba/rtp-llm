load("//:def.bzl", "copts", "cuda_copts", "torch_deps")
load("//bazel:arch_select.bzl", "th_transformer_so")

config_setting(
    name = "use_cuda12",
    values = {"define": "use_cuda12=true"},
)

cc_library(
    name = "th_op_hdrs",
    hdrs = glob([
        "src/fastertransformer/th_op/**/*.h",
    ]),
)

cc_library(
    name = "th_transformer_lib",
    srcs = glob([
        "src/fastertransformer/th_op/th_utils.cc",
        "src/fastertransformer/th_op/GptInitParameter.cc",
        "src/fastertransformer/th_op/common/*.cc",
        "src/fastertransformer/th_op/multi_gpu_gpt/*.cc",
    ]),
    deps = [
    	":th_op_hdrs",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/models:models",
        "//src/fastertransformer/utils:torch_utils",
    ],
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "th_transformer",
    deps = [
        "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        "//3rdparty/flash_attention2:flash_attention2_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        ":th_transformer_lib"
    ],
    copts = copts(),
    linkshared = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_utils",
    srcs = [
        "src/fastertransformer/th_op/th_utils.cc",
        "src/fastertransformer/th_op/GptInitParameter.cc",
    ],
    hdrs = [
        "src/fastertransformer/th_op/th_utils.h",
        "src/fastertransformer/th_op/GptInitParameter.h",
        "src/fastertransformer/th_op/GptCommonInputs.h",
    ],
    deps = [
        "//src/fastertransformer/utils:torch_utils",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/kernels:kernels",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

py_runtime(
    name = "python310",
    interpreter_path = "/opt/conda310/bin/python",
    python_version = "PY3",
    stub_shebang = "#!/opt/conda310/bin/python"
)

cc_binary(
    name = "kernel_unittest",
    srcs = glob([
        "tests/layernorm/*.cpp",
        "tests/logn_attention/*.cpp",
        "tests/rotary_embedding/*.cpp",
    ]),
    deps = [
        "//tests:test_ops",
    ],
    copts = cuda_copts(),
    linkshared = 1,
    visibility = ["//visibility:public"],
)
