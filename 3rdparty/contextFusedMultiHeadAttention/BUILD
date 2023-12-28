load("//:def.bzl", "rpm_library", "torch_deps", "copts", "cuda_copts",)
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trt_fmha_impl",
    srcs = glob([
        "fmhaRunner.cpp",
        "**/*.cpp"
    ]),
    hdrs = glob([
        "*.h",
        "*.cuh",
    ]),    
    deps = [
        "@cutlass//:cutlass",
        ":trt_fmha_header",
        "//3rdparty:cuda_driver",
        "//src/fastertransformer/kernels:kernels",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",        
    ],
    copts = cuda_copts(),
    visibility = ["//:__subpackages__"],
    alwayslink = True,
)


cc_library(
    name = "trt_fmha_header",
    hdrs = glob([
        "*.h",
        "**/*.h"
    ]),    
    deps = [        
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",    
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],    
)