load("//:def.bzl", "copts", "cuda_copts", "torch_deps")
load("//bazel:arch_select.bzl", "th_transformer_so")
load("//bazel:arch_select.bzl", "cutlass_kernels_interface")

cutlass_kernels_interface()

config_setting(
    name = "using_cuda",
    values = {"define": "using_cuda=true"},
)

config_setting(
    name = "use_cuda12",
    values = {"define": "use_cuda12=true"},
)

cc_library(
    name = "gpt_init_params_hdr",
    hdrs = [
        "src/fastertransformer/th_op/GptInitParameter.h"
    ],
    deps = [
        "//src/fastertransformer/utils:utils",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
)

# NOTE: This target is defined here but not used here.
# for libth_transformer.so, GptInitParameter.cc must be compiled together with `th_op/multi_gpu_gpt/*.cc`
# in a single target, otherwise torch throws an error of
# `Type c10::intrusive_ptr<GptInitParameter> could not be converted to any of the known types.`
# This is due to GptInitParameter is referenced before it's registered,
# which might because the compiled symbols does not load in expected order according to dependency.
cc_library(
    name = "gpt_init_params",
    srcs = [
        "src/fastertransformer/th_op/GptInitParameter.cc"
    ],
    deps = [
        ":gpt_init_params_hdr",
    ],
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_op_hdrs",
    hdrs = glob([
        "src/fastertransformer/th_op/**/*.h",
    ], exclude = [
        "src/fastertransformer/th_op/GptInitParameter.h"
    ]),
)

cc_library(
    name = "th_transformer_lib",
    srcs = glob([
        "src/fastertransformer/th_op/th_utils.cc",
        "src/fastertransformer/th_op/common/*.cc",
        "src/fastertransformer/th_op/multi_gpu_gpt/*.cc",
        "src/fastertransformer/th_op/GptInitParameter.cc"
    ]),
    deps = [
        ":gpt_init_params_hdr",
    	":th_op_hdrs",
        "//src/fastertransformer/cuda:allocator_torch",
        "//src/fastertransformer/layers:layers",
        "//src/fastertransformer/models:models",
        "//src/fastertransformer/utils:utils",
    ],
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "th_transformer",
    deps = [
        "cutlass_kernels_interface",
        "//3rdparty/flash_attention2:flash_attention2_impl",
        "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
        ":th_transformer_lib",
        ":gpt_init_params_hdr",
    ],
    copts = copts(),
    linkshared = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_utils",
    srcs = [
        "src/fastertransformer/th_op/th_utils.cc",
    ],
    hdrs = [
        "src/fastertransformer/th_op/th_utils.h",
        "src/fastertransformer/th_op/GptCommonInputs.h",
    ],
    deps = [
        "//src/fastertransformer/cuda:allocator_torch",
        "//src/fastertransformer/cuda:cuda",
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
