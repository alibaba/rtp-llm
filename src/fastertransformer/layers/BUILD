load("//:def.bzl", "copts", "cuda_copts")

cc_library(
    name = "layers",
    srcs = glob([
        "**/*.cu",
        "**/*.cc",
    ], exclude=[
        "attention_layers_fp8/*.cc",
        "FfnFP8Layer.cc",
        "TensorParallelGeluFfnFP8Layer.cc",
    ]),
    hdrs = glob([
        "**/*.h",
    ]),
    deps = [
        "//src/fastertransformer/trt_plugins:trt_plugins",
        "//src/fastertransformer/cutlass:cutlass_interface",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/cuda:cuda",
        "//src/fastertransformer/kernels:kernels",
        "//3rdparty:cuda_driver",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
