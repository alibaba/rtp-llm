genrule(
    name = "cpp_libraries",
    srcs = glob([
        "**/*",
    ]) + ["@composable_kernel_archive//:config_h",],
    outs = [
        "aiter/jit/libmodule_aiter_enum.so",
        "aiter/jit/libmodule_custom_all_reduce.so",
        # "csrc/cpp_itfs/mla/libasm_mla_decode_fwd_torch.so",
        # "aiter/jit/libmodule_attention.so",
        # "aiter/jit/libmodule_norm.so",
        # "aiter/jit/libmodule_cache.so",
        # "aiter/jit/libmodule_mha_fwd.so",
        "aiter/jit/libmodule_quant.so",
        "aiter/jit/libmodule_gemm_a8w8_blockscale.so",
        "aiter/jit/libmodule_moe_sorting.so",
        "aiter/jit/libmodule_moe_asm.so",
        "aiter/jit/libmodule_pa.so",
        "aiter/jit/libmodule_gemm_a8w8_bpreshuffle.so",
        "aiter/jit/libmodule_moe.so",
        "aiter/jit/libmodule_activation.so",
    ],
    cmd = """
        awk '{gsub(/^#cmakedefine DTYPES \"@DTYPES@\"/, "/* #undef DTYPES*/");
              gsub(/^#cmakedefine CK_ENABLE_ALL_DTYPES @CK_ENABLE_ALL_DTYPES@/, "#define CK_ENABLE_ALL_DTYPES ON");
              gsub(/^#cmakedefine CK_ENABLE_INT8 @CK_ENABLE_INT8@/, "/* #undef CK_ENABLE_INT8*/");
              gsub(/^#cmakedefine CK_ENABLE_FP8 @CK_ENABLE_FP8@/, "/* #undef CK_ENABLE_FP8*/");
              gsub(/^#cmakedefine CK_ENABLE_BF8 @CK_ENABLE_BF8@/, "/* #undef CK_ENABLE_BF8*/");
              gsub(/^#cmakedefine CK_ENABLE_FP16 @CK_ENABLE_FP16@/, "/* #undef CK_ENABLE_FP16*/");
              gsub(/^#cmakedefine CK_ENABLE_BF16 @CK_ENABLE_BF16@/, "/* #undef CK_ENABLE_BF16*/");
              gsub(/^#cmakedefine CK_ENABLE_FP32 @CK_ENABLE_FP32@/, "/* #undef CK_ENABLE_FP32*/");
              gsub(/^#cmakedefine CK_ENABLE_FP64 @CK_ENABLE_FP64@/, "/* #undef CK_ENABLE_FP64*/");
              gsub(/^#cmakedefine CK_ENABLE_DL_KERNELS @CK_ENABLE_DL_KERNELS@/, "/* #undef CK_ENABLE_DL_KERNELS*/");
              gsub(/^#cmakedefine CK_ENABLE_DPP_KERNELS @CK_ENABLE_DPP_KERNELS@/, "/* #undef CK_ENABLE_DPP_KERNELS*/");
              gsub(/^#cmakedefine CK_ENABLE_INSTANCES_ONLY @CK_ENABLE_INSTANCES_ONLY@/, "/* #undef CK_ENABLE_INSTANCES_ONLY*/");
              gsub(/^#cmakedefine CK_USE_XDL @CK_USE_XDL@/, "#define CK_USE_XDL ON");
              gsub(/^#cmakedefine CK_USE_WMMA @CK_USE_WMMA@/, "/* #undef CK_USE_WMMA*/");
              gsub(/^#cmakedefine CK_USE_GFX94 @CK_USE_GFX94@/, "/* #undef CK_USE_GFX94*/");
              gsub(/^#cmakedefine CK_USE_OCP_FP8 @CK_USE_OCP_FP8@/, "/* #undef CK_USE_OCP_FP8*/");
              gsub(/^#cmakedefine CK_USE_FNUZ_FP8 @CK_USE_FNUZ_FP8@/, "/* #undef CK_USE_FNUZ_FP8*/");
              gsub(/^#cmakedefine CK_USE_FP8_ON_UNSUPPORTED_ARCH @CK_USE_FP8_ON_UNSUPPORTED_ARCH@/, "/* #undef CK_USE_FP8_ON_UNSUPPORTED_ARCH*/");
              gsub(/^#cmakedefine CK_USE_NATIVE_MX_SUPPORT @CK_USE_NATIVE_MX_SUPPORT@/, "/* #undef CK_USE_NATIVE_MX_SUPPORT*/");
              gsub(/^#cmakedefine/, "//cmakedefine");print;}' external/aiter_src/3rdparty/composable_kernel/include/ck/config.h.in > external/aiter_src/3rdparty/composable_kernel/include/ck/config.h;
        cd external/aiter_src;
        find . -name lock | xargs rm -f;
        /opt/conda310/bin/python -m pip install -r requirements.txt -i https://artifacts.antgroup-inc.cn/simple/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/py-central/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/pytorch/ --extra-index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion --trusted-host=artlab.alibaba-inc.com;
        /opt/conda310/bin/python -m pip install ninja -i https://artifacts.antgroup-inc.cn/simple/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/py-central/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/pytorch/ --extra-index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion --trusted-host=artlab.alibaba-inc.com;
        /opt/conda310/bin/python -m pip install packaging -i https://artifacts.antgroup-inc.cn/simple/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/py-central/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/pytorch/ --extra-index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion --trusted-host=artlab.alibaba-inc.com;
        GPU_ARCHS=gfx942 ROCM_HOME=/opt/rocm LD_LIBRARY_PATH=/opt/amdgpu/lib64 PATH=/opt/rocm/bin:/opt/conda310/bin:$$PATH /opt/conda310/bin/python build_aiter_module.py;
        
        cd ../..;
        cp external/aiter_src/aiter/jit/module_aiter_enum.so $(location aiter/jit/libmodule_aiter_enum.so);
        cp external/aiter_src/aiter/jit/module_custom_all_reduce.so $(location aiter/jit/libmodule_custom_all_reduce.so);
        cp external/aiter_src/aiter/jit/module_quant.so $(location aiter/jit/libmodule_quant.so);
        cp external/aiter_src/aiter/jit/module_moe_sorting.so $(location aiter/jit/libmodule_moe_sorting.so);
        cp external/aiter_src/aiter/jit/module_moe_asm.so $(location aiter/jit/libmodule_moe_asm.so);
        cp external/aiter_src/aiter/jit/module_moe.so $(location aiter/jit/libmodule_moe.so);
        cp external/aiter_src/aiter/jit/module_gemm_a8w8_blockscale.so $(location aiter/jit/libmodule_gemm_a8w8_blockscale.so);
        cp external/aiter_src/aiter/jit/module_pa.so $(location aiter/jit/libmodule_pa.so);
        cp external/aiter_src/aiter/jit/module_gemm_a8w8_bpreshuffle.so $(location aiter/jit/libmodule_gemm_a8w8_bpreshuffle.so);
        cp external/aiter_src/aiter/jit/module_activation.so $(location aiter/jit/libmodule_activation.so);
    """,
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_custom_all_reduce",
    srcs = ["aiter/jit/libmodule_custom_all_reduce.so"],
    hdrs = ["csrc/include/custom_all_reduce.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

# cc_library(
#     name = "decode_mla",
#     srcs = ["csrc/cpp_itfs/mla/libasm_mla_decode_fwd_torch.so"],
#     hdrs = ["csrc/cpp_itfs/mla/asm_mla_decode_fwd_torch.h"],
#     deps = [":cpp_libraries"],
#     copts = [],
# #     strip_include_prefix = "csrc/cpp_itfs/",
#     visibility = ["//visibility:public"],
#     tags = ["rocm","local"],
# )

# cc_library(
#     name = "module_mha_fwd",
#     srcs = ["aiter/jit/libmodule_mha_fwd.so"],
#     hdrs = ["csrc/include/mha_fwd.h"],
#     deps = [":cpp_libraries"],
#     copts = [],
# #     strip_include_prefix = "csrc/include/",
#     visibility = ["//visibility:public"],
# )

cc_library(
    name = "module_aiter_enum",
    srcs = ["aiter/jit/libmodule_aiter_enum.so"],
    hdrs = ["csrc/include/aiter_enum.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_quant",
    srcs = ["aiter/jit/libmodule_quant.so"],
    hdrs = ["csrc/include/quant.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_gemm_a8w8_blockscale",
    srcs = ["aiter/jit/libmodule_gemm_a8w8_blockscale.so"],
    hdrs = ["csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/ck_gemm_a8w8_blockscale/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_moe",
    srcs = ["aiter/jit/libmodule_moe.so"],
    hdrs = ["csrc/include/moe_ck.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_gemm_a8w8_bpreshuffle",
    srcs = ["aiter/jit/libmodule_gemm_a8w8_bpreshuffle.so"],
    hdrs = ["csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/ck_gemm_a8w8_bpreshuffle/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_moe_sorting",
    srcs = ["aiter/jit/libmodule_moe_sorting.so"],
    hdrs = ["csrc/include/moe_sorting.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_moe_asm",
    srcs = ["aiter/jit/libmodule_moe_asm.so"],
    hdrs = ["csrc/include/moe_op.h", "csrc/include/aiter_enum.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_pa",
    srcs = ["aiter/jit/libmodule_pa.so"],
    hdrs = ["csrc/include/attention.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_activation",
    srcs = ["aiter/jit/libmodule_activation.so"],
    hdrs = ["csrc/include/activation.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)
