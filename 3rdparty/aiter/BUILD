load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
)

genrule(
    name = "config_h",
    srcs = [
        "3rdparty/composable_kernel/include/ck/config.h.in",
    ],
    outs = [
        "3rdparty/composable_kernel/include/ck/config.h",
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
            gsub(/^#cmakedefine CK_USE_WMMA @CK_USE_WMMA@/, "/* #undef CK_USE_WMMA*/");
            gsub(/^#cmakedefine/, "//cmakedefine");print;}' $(<) > $(@)
    """,
)

cc_library(
    name = "ck_headers_real",
    hdrs = glob([
        "3rdparty/composable_kernel/include/**/*.h",
        "3rdparty/composable_kernel/include/**/*.inc",
        "3rdparty/composable_kernel/include/**/*.hpp",
    ], exclude = ["3rdparty/composable_kernel/include/rapidjson/**"]),
    copts = rocm_default_copts() + ["-std=c++20"],
    strip_include_prefix = "3rdparty/composable_kernel/include",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
        ":config_h"
    ],
)

cc_library(
    name = "ck_library_headers",
    srcs = glob(["3rdparty/composable_kernel/library/src/utility/**/*.cpp"]),
    hdrs = glob([
        "3rdparty/composable_kernel/library/include/**/*.h",
        "3rdparty/composable_kernel/library/include/**/*.inc",
        "3rdparty/composable_kernel/library/include/**/*.hpp",
    ]),
    strip_include_prefix = "3rdparty/composable_kernel/library/include",
    copts = rocm_default_copts() + ["-std=c++20"],
    deps = [
        ":ck_headers_real",
    ],
)

cc_library(
    name = "ck_fmha_example_headers",
    hdrs = glob([
        "3rdparty/composable_kernel/example/ck_tile/01_fmha/*.hpp",
    ]),
    copts = rocm_default_copts() + ["-std=c++20"],
    deps = [
        ":ck_headers_real",
        ":ck_library_headers",
    ],
    strip_include_prefix = "3rdparty/composable_kernel/example/ck_tile/01_fmha",
    visibility = ["//visibility:public"],
)

genrule(
    name = "cpp_libraries",
    srcs = glob([
        "**/*",
    ]) + [":config_h"],
    outs = [
        "aiter/jit/libmodule_aiter_enum.so",
        "aiter/jit/libmodule_custom_all_reduce.so",
        "aiter/jit/libmodule_quick_all_reduce.so",
        "aiter/jit/libmodule_norm.so",
        "aiter/jit/libmodule_mha_fwd.so",
        "aiter/jit/libmodule_fmha_v3_varlen_fwd.so",
        "aiter/jit/libmodule_quant.so",
        "aiter/jit/libmodule_gemm_a8w8_blockscale.so",
        "aiter/jit/libmodule_moe_sorting.so",
        "aiter/jit/libmodule_moe_asm.so",
        "aiter/jit/libmodule_pa.so",
        "aiter/jit/libmodule_attention_asm.so",
        "aiter/jit/libmodule_gemm_a8w8_bpreshuffle.so",
        "aiter/jit/libmodule_activation.so",
        "aiter/jit/libmodule_rmsnorm.so",
        "aiter/jit/libmodule_smoothquant.so",
        "aiter/jit/libmodule_gemm_a8w8.so",
        "aiter/jit/libmodule_moe_ck2stages.so",
        "aiter/jit/libmodule_m_grouped_gemm.so",
        "aiter/jit/libmodule_quick_all_reduce.so",
    ],
    cmd = """
        cd external/aiter_src;
        export PYTHONPATH=$${PWD}:$${PYTHONPATH:-};
        find . -name lock_* | xargs rm -f;
        /opt/conda310/bin/python -m pip install -r requirements.txt -i https://artifacts.antgroup-inc.cn/simple/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/py-central/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/pytorch/ --extra-index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion --trusted-host=artlab.alibaba-inc.com;
        /opt/conda310/bin/python -m pip install ninja -i https://artifacts.antgroup-inc.cn/simple/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/py-central/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/pytorch/ --extra-index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion --trusted-host=artlab.alibaba-inc.com;
        /opt/conda310/bin/python -m pip install packaging -i https://artifacts.antgroup-inc.cn/simple/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/py-central/ --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/pytorch/ --extra-index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion --trusted-host=artlab.alibaba-inc.com;
        LD_LIBRARY_PATH=/opt/amdgpu/lib64:/opt/rocm-6.4.3/lib/llvm/lib/:$$LD_LIBRARY_PATH AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx942 ROCM_HOME=/opt/rocm PATH=/opt/rocm/bin:/opt/conda310/bin:$$PATH /opt/conda310/bin/python build_aiter_module.py;
        
        cd ../..;
        cp external/aiter_src/aiter/jit/module_aiter_enum.so $(location aiter/jit/libmodule_aiter_enum.so);
        cp external/aiter_src/aiter/jit/module_custom_all_reduce.so $(location aiter/jit/libmodule_custom_all_reduce.so);
        cp external/aiter_src/aiter/jit/module_quick_all_reduce.so $(location aiter/jit/libmodule_quick_all_reduce.so);
        cp external/aiter_src/aiter/jit/module_quant.so $(location aiter/jit/libmodule_quant.so);
        cp external/aiter_src/aiter/jit/module_smoothquant.so $(location aiter/jit/libmodule_smoothquant.so);
        cp external/aiter_src/aiter/jit/module_moe_sorting.so $(location aiter/jit/libmodule_moe_sorting.so);
        cp external/aiter_src/aiter/jit/module_moe_asm.so $(location aiter/jit/libmodule_moe_asm.so);
        cp external/aiter_src/aiter/jit/module_gemm_a8w8_blockscale.so $(location aiter/jit/libmodule_gemm_a8w8_blockscale.so);
        cp external/aiter_src/aiter/jit/module_pa.so $(location aiter/jit/libmodule_pa.so);
        cp external/aiter_src/aiter/jit/module_attention_asm.so $(location aiter/jit/libmodule_attention_asm.so);
        cp external/aiter_src/aiter/jit/module_gemm_a8w8_bpreshuffle.so $(location aiter/jit/libmodule_gemm_a8w8_bpreshuffle.so);
        cp external/aiter_src/aiter/jit/module_activation.so $(location aiter/jit/libmodule_activation.so);
        cp external/aiter_src/aiter/jit/module_m_grouped_gemm.so $(location aiter/jit/libmodule_m_grouped_gemm.so);
        cp external/aiter_src/aiter/jit/module_norm.so $(location aiter/jit/libmodule_norm.so);
        cp external/aiter_src/aiter/jit/module_rmsnorm.so $(location aiter/jit/libmodule_rmsnorm.so);
        cp external/aiter_src/aiter/jit/module_mha_fwd.so $(location aiter/jit/libmodule_mha_fwd.so);
        cp external/aiter_src/aiter/jit/module_fmha_v3_varlen_fwd.so $(location aiter/jit/libmodule_fmha_v3_varlen_fwd.so);
        cp external/aiter_src/aiter/jit/module_gemm_a8w8.so $(location aiter/jit/libmodule_gemm_a8w8.so);
        cp external/aiter_src/aiter/jit/module_moe_ck2stages.so $(location aiter/jit/libmodule_moe_ck2stages.so);
        cp external/aiter_src/aiter/jit/module_quick_all_reduce.so $(location aiter/jit/libmodule_quick_all_reduce.so);
    """,
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

# Library for py_itfs_common.h header
cc_library(
    name = "py_itfs_common_headers",
    hdrs = ["csrc/include/py_itfs_common.h"],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm"],
)

cc_library(
    name = "ck_deepgemm_example_headers",
    hdrs = glob([
        "3rdparty/composable_kernel/example/ck_tile/18_flatmm/*.hpp",
    ]),
    copts = rocm_default_copts() + ["-std=c++20"],
    deps = [
        ":ck_headers_real",
        ":ck_library_headers",
        ":py_itfs_common_headers",
    ],
    strip_include_prefix = "3rdparty/composable_kernel/example/ck_tile/18_flatmm",
    visibility = ["//visibility:public"],
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

cc_library(
    name = "module_quick_all_reduce",
    srcs = ["aiter/jit/libmodule_quick_all_reduce.so"],
    hdrs = ["csrc/include/quick_all_reduce.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_mha_fwd",
    srcs = ["aiter/jit/libmodule_mha_fwd.so"],
    hdrs = ["csrc/include/mha_fwd.h", "csrc/include/aiter_hip_common.h"],
    deps = [":cpp_libraries", ":ck_fmha_example_headers"],
    copts = ["-std=c++20"],
    linkopts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_fmha_v3_varlen_fwd",
    srcs = ["aiter/jit/libmodule_fmha_v3_varlen_fwd.so"],
    hdrs = ["csrc/include/torch/mha_v3_varlen_fwd.h"],
    deps = [":cpp_libraries", ":ck_fmha_example_headers"],
    copts = ["-std=c++20"],
    linkopts = [],
    strip_include_prefix = "csrc/include/torch/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

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
    name = "module_smoothquant",
    srcs = ["aiter/jit/libmodule_smoothquant.so"],
    hdrs = ["csrc/include/smoothquant.h"],
    deps = [":cpp_libraries"],
    copts = [],
    linkopts = [],
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
    srcs = [
        "aiter/jit/libmodule_pa.so", 
        "aiter/jit/libmodule_attention_asm.so"
    ],
    hdrs = [
        "csrc/include/attention.h",
        "csrc/include/attention_asm.h"
    ],
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

cc_library(
    name = "module_rmsnorm",
    srcs = ["aiter/jit/libmodule_rmsnorm.so"],
    hdrs = ["csrc/include/rmsnorm.h"],
    deps = [":cpp_libraries"],
    copts = [],
    linkopts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_norm",
    srcs = ["aiter/jit/libmodule_norm.so"],
    hdrs = ["csrc/include/norm.h"],
    deps = [":cpp_libraries"],
    copts = [],
    linkopts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_gemm_a8w8",
    srcs = ["aiter/jit/libmodule_gemm_a8w8.so"],
    hdrs = ["csrc/ck_gemm_a8w8/include/gemm_a8w8.h"],
    deps = [":cpp_libraries"],
    copts = [],
    linkopts = [],
    strip_include_prefix = "csrc/ck_gemm_a8w8/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)

cc_library(
    name = "module_moe_ck2stages",
    srcs = [
        "aiter/jit/libmodule_moe_ck2stages.so"
    ],
    hdrs = ["csrc/include/moe_ck.h"],
    deps = [":cpp_libraries"],
    copts = [],
    linkopts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)
 
cc_library(
    name = "module_m_grouped_gemm",
    srcs = ["aiter/jit/module_deepgemm.so"],
    hdrs = ["csrc/ck_deepgemm/include/deepgemm.h"],
    deps = [
        ":aiter_so",
        ":aiter_headers",
        ":ck_deepgemm_example_headers"
    ],
    copts = [],
    strip_include_prefix = "csrc/ck_deepgemm/include",
    visibility = ["//visibility:public"],
    tags = ["rocm"],
)

cc_library(
    name = "module_quick_all_reduce",
    srcs = ["aiter/jit/libmodule_quick_all_reduce.so"],
    hdrs = ["csrc/include/quick_all_reduce.h"],
    deps = [":cpp_libraries"],
    copts = [],
    strip_include_prefix = "csrc/include/",
    visibility = ["//visibility:public"],
    tags = ["rocm","local"],
)
