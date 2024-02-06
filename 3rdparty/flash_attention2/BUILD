load("//:def.bzl", "rpm_library", "torch_deps", "copts", "cuda_copts",)
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "flash_attention2_impl",
    srcs = glob([
        "flash_fwd_hdim128_bf16_sm80.cu",
        "flash_fwd_hdim128_fp16_sm80.cu",
        "flash_fwd_hdim96_bf16_sm80.cu",
        "flash_fwd_hdim96_fp16_sm80.cu",
        "flash_fwd_hdim64_bf16_sm80.cu",
        "flash_fwd_hdim64_fp16_sm80.cu",
        "*.cpp"
    ]),
    hdrs = glob([
        "*.h",
        "*.cuh",
    ]),    
    deps = [
        "@cutlass//:cutlass",
        ":flash_attention2_header",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",        
    ],
    copts = cuda_copts(),
    visibility = ["//:__subpackages__"],
    alwayslink = True,
)


cc_library(
    name = "flash_attention2_header",
    hdrs = glob([
        "flash.h",
    ]),    
    deps = [        
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],    
)