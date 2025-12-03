load("//bazel:arch_select.bzl", "kernel_so_deps", "copy_all_so")
load("//bazel:defs.bzl", "copy_files")
package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_all_so()

# copy 3fs libs
copy_files(
    name = "copy_hf3fs_libs",
    srcs = select({
        "//:using_3fs": ["//3rdparty/3fs:hf3fs_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "copy_aiter",
    srcs = select({
        "@//:using_rocm_with_aiter_src": [":aiter_src_copy",],
        "@//:using_aiter_src": [":aiter_src_copy",],
        "//conditions:default": [":aiter_copy"],
    }),
    visibility = ["//visibility:public"],
)

copy_files(
    name = "copy_acclep_libs",
    srcs = select({
        "//rtp_llm/cpp/devices/cuda_impl:use_accl_ep": ["//3rdparty/accl_ep:accl_ep_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

genrule(
    name = "aiter_src_copy",
    srcs = ["@aiter_src//:cpp_libraries", "@deep_ep_rocm//:cpp_libraries"],
    outs = [
        "libmodule_aiter_enum.so",
        "libmodule_custom_all_reduce.so",
        "libmodule_quick_all_reduce.so",
        "libmodule_moe_sorting.so",
        "libmodule_moe_asm.so",
        "libmodule_gemm_a8w8_bpreshuffle.so",
        "libmodule_gemm_a8w8_blockscale.so",
        "libmodule_quant.so",
        "libmodule_smoothquant.so",
        "libmodule_pa.so",
        "libmodule_activation.so",
        "libmodule_m_grouped_gemm.so",
        "libdeep_ep_rocm.so",
        "libmodule_attention_asm.so",
        "libmodule_mha_fwd.so",
        "libmodule_fmha_v3_varlen_fwd.so",
        "libmodule_norm.so",
        "libmodule_rmsnorm.so",
        "libmodule_gemm_a8w8.so",
        "libmodule_moe_ck2stages.so",
    ],
    cmd = """
       cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_aiter_enum.so $(location libmodule_aiter_enum.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_custom_all_reduce.so $(location libmodule_custom_all_reduce.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_quick_all_reduce.so $(location libmodule_quick_all_reduce.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_moe_sorting.so $(location libmodule_moe_sorting.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_moe_asm.so $(location libmodule_moe_asm.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_gemm_a8w8_bpreshuffle.so $(location libmodule_gemm_a8w8_bpreshuffle.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_gemm_a8w8_blockscale.so $(location libmodule_gemm_a8w8_blockscale.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_quant.so $(location libmodule_quant.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_smoothquant.so $(location libmodule_smoothquant.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_pa.so $(location libmodule_pa.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_activation.so $(location libmodule_activation.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_attention_asm.so $(location libmodule_attention_asm.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_mha_fwd.so $(location libmodule_mha_fwd.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_fmha_v3_varlen_fwd.so $(location libmodule_fmha_v3_varlen_fwd.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_norm.so $(location libmodule_norm.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_rmsnorm.so $(location libmodule_rmsnorm.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_gemm_a8w8.so $(location libmodule_gemm_a8w8.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_moe_ck2stages.so $(location libmodule_moe_ck2stages.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_m_grouped_gemm.so $(location libmodule_m_grouped_gemm.so);
cp ./bazel-out/k8-opt/bin/external/deep_ep_rocm/libdeep_ep_rocm.so $(location libdeep_ep_rocm.so);
""",
tags = ["rocm", "local"],
)

genrule(
    name = "aiter_copy",
    srcs = ["@aiter//:aiter_so", "@deep_ep_rocm//:cpp_libraries"],
    outs = [
        "module_aiter_enum.so",
        "module_custom_all_reduce.so",
        "module_quick_all_reduce.so",
        "module_moe_sorting.so",
        "module_moe_asm.so",
        "module_gemm_a8w8_bpreshuffle.so",
        "module_gemm_a8w8_blockscale.so",
        "module_quant.so",
        "module_smoothquant.so",
        "module_pa.so",
        "module_activation.so",
        "module_attention_asm.so",
        "module_mha_fwd.so",
        "module_fmha_v3_varlen_fwd.so",
        "module_norm.so",
        "module_rmsnorm.so",
        "module_gemm_a8w8.so",
        "module_moe_ck2stages.so",
        "module_deepgemm.so",
        "deep_ep_rocm.so",
    ],
    cmd = """
        cp external/aiter/aiter/jit/module_aiter_enum.so $(location module_aiter_enum.so);
        cp external/aiter/aiter/jit/module_custom_all_reduce.so $(location module_custom_all_reduce.so);
        cp external/aiter/aiter/jit/module_quick_all_reduce.so $(location module_quick_all_reduce.so);
        cp external/aiter/aiter/jit/module_moe_sorting.so $(location module_moe_sorting.so);
        cp external/aiter/aiter/jit/module_moe_asm.so $(location module_moe_asm.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8_bpreshuffle.so $(location module_gemm_a8w8_bpreshuffle.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8_blockscale.so $(location module_gemm_a8w8_blockscale.so);
        cp external/aiter/aiter/jit/module_quant.so $(location module_quant.so);
        cp external/aiter/aiter/jit/module_smoothquant.so $(location module_smoothquant.so);
        cp external/aiter/aiter/jit/module_pa.so $(location module_pa.so);
        cp external/aiter/aiter/jit/module_activation.so $(location module_activation.so);
        cp external/aiter/aiter/jit/module_attention_asm.so $(location module_attention_asm.so);
        cp external/aiter/aiter/jit/module_mha_fwd.so $(location module_mha_fwd.so);
        cp external/aiter/aiter/jit/module_fmha_v3_varlen_fwd.so $(location module_fmha_v3_varlen_fwd.so);
        cp external/aiter/aiter/jit/module_norm.so $(location module_norm.so);
        cp external/aiter/aiter/jit/module_rmsnorm.so $(location module_rmsnorm.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8.so $(location module_gemm_a8w8.so);
        cp external/aiter/aiter/jit/module_moe_ck2stages.so $(location module_moe_ck2stages.so);
        cp external/aiter/aiter/jit/module_deepgemm.so $(location module_deepgemm.so);
        cp ./bazel-out/k8-opt/bin/external/deep_ep_rocm/libdeep_ep_rocm.so $(location deep_ep_rocm.so);
""",
    tags = ["rocm", "local"],
)

filegroup(
    name = "frontend_libs",
    srcs = [],
    data = [
        ":libth_transformer_config_so"
    ]
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":copy_hf3fs_libs",
        ":copy_aiter",
        ":copy_acclep_libs",
    ] + kernel_so_deps()
)

filegroup(
    name = "whl_package_libs",
    srcs = [],
    data = [
        ":libth_transformer_config_so",
        ":libth_transformer_so",
        ":librtp_compute_ops_so",
    ]
)
