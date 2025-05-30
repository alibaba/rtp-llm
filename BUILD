load("//:def.bzl", "copts", "cuda_copts")
load("//bazel:arch_select.bzl", "torch_deps")

config_setting(
    name = "enable_triton",
    values = {"define": "enable_triton=true"},
)

config_setting(
    name = "using_cuda",
    values = {"define": "using_cuda=true"},
)

config_setting(
    name = "using_cuda12",
    values = {"define": "using_cuda12=true"},
)

config_setting(
    name = "using_cuda11",
    values = {"define": "using_cuda11=true"},
)

config_setting(
    name = "using_rocm",
    values = {"define": "using_rocm=true"},
)

config_setting(
    name = "using_arm",
    values = {"define": "using_arm=true"},
)

config_setting(
    name = "using_cpu",
    values = {"define": "using_cpu=true"},
)

config_setting(
    name = "xft_use_icx",
    values = {"define": "xft_use_icx=true"},
)


cc_library(
    name = "gpt_init_params",
    srcs = [
        "//maga_transformer/cpp:th_op/GptInitParameter.cc"
    ],
    hdrs = [
        "//maga_transformer/cpp:th_op/GptInitParameter.h",
        "//maga_transformer/cpp:th_op/GptInitParameterRegister.h",
    ],
    deps = [
        "//maga_transformer/cpp:utils",
	    "//maga_transformer/cpp/core:types"
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "th_op_hdrs_files",
    srcs = [
        "//maga_transformer/cpp:th_op/common/NcclOp.h",
        "//maga_transformer/cpp:th_op/common/InitEngineOps.h",
        "//maga_transformer/cpp:th_op/multi_gpu_gpt/EmbeddingHandlerOp.h",
        "//maga_transformer/cpp:th_op/multi_gpu_gpt/RtpEmbeddingOp.h",
        "//maga_transformer/cpp:th_op/multi_gpu_gpt/RtpLLMOp.h",
    ],
)

cc_library(
    name = "th_op_hdrs",
    hdrs = [
        ":th_op_hdrs_files",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "th_transformer_lib_files",
    srcs = [
        "//maga_transformer/cpp:th_op/GptInitParameter.cc",
        "//maga_transformer/cpp:th_op/init.cc",
        "//maga_transformer/cpp:th_op/common/InitEngineOps.cc",
        "//maga_transformer/cpp:th_op/multi_gpu_gpt/RtpEmbeddingOp.cc",
        "//maga_transformer/cpp:th_op/multi_gpu_gpt/EmbeddingHandlerOp.cc",
        "//maga_transformer/cpp:th_op/multi_gpu_gpt/RtpLLMOp.cc",
    ] + select({
        "@//:using_cuda": [
            "//maga_transformer/cpp:th_op/common/NcclOp.cc",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_lib",
    srcs = [
        ":th_transformer_lib_files"
    ],
    deps = [
        ":gpt_init_params",
    	":th_op_hdrs",
        "//maga_transformer/cpp:utils",
        "//maga_transformer/cpp/devices:device_py_export",
        "//maga_transformer/cpp/devices:devices_base",
        "//maga_transformer/cpp:http_api_server",
        "//maga_transformer/cpp:model_rpc_server",
        "@grpc//:grpc++",
    ] + select({
        "@//:using_cuda": [
            "//maga_transformer/cpp/cuda:allocator_torch",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_transformer_gpu",
    srcs = select({
        "@//:using_cuda11": [
            "//maga_transformer/cpp:th_op/common/CutlassConfigOps.cc",
        ],
        "@//:using_cuda12": [
            "//maga_transformer/cpp:th_op/common/CutlassConfigOps.cc",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":gpt_init_params",
    	":th_op_hdrs",
        "//maga_transformer/cpp:utils",
        "//maga_transformer/cpp:model_rpc_server",
        "@grpc//:grpc++",
    ] + select({
        "@//:using_cuda": [
            "//maga_transformer/cpp/cuda:allocator_torch",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "th_transformer",
    deps = [
        ":th_transformer_lib",
        ":gpt_init_params",
    ] + select({
        "@//:using_cuda": [
            ":th_transformer_gpu",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    linkshared = 1,
    linkopts = [
        "-Wl,-rpath='$$ORIGIN'",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "th_utils",
    hdrs = [
        "//maga_transformer/cpp:th_op/th_utils.h",
    ],
    deps = [
        "//maga_transformer/cpp:utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

py_runtime(
    name = "python310",
    interpreter_path = "/opt/conda310/bin/python",
    visibility = ["//visibility:public"],
    python_version = "PY3",
    stub_shebang = "#!/opt/conda310/bin/python"
)
