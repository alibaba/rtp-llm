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

genrule(
    name = "aiter_copy",
    srcs = ["@aiter//:aiter_so","@composable_kernel_archive//:ck_fmha_rmsnorm2d_libraries"],
    outs = [
        "module_aiter_enum.so",
        "module_custom_all_reduce.so",
        "module_moe_sorting.so",
        "module_moe_asm.so",
        "module_moe.so",
        "module_gemm_a8w8_bpreshuffle.so",
        "module_gemm_a8w8_blockscale.so",
        "module_quant.so",
        "module_pa.so",
        "libtile_rmsnorm2d_fwd.so",
        "libtile_example_fmha_fwd.so",
        "module_activation.so",
    ],
    cmd = """
        cp external/aiter/aiter/jit/module_aiter_enum.so $(location module_aiter_enum.so);
        cp external/aiter/aiter/jit/module_custom_all_reduce.so $(location module_custom_all_reduce.so);
        cp external/aiter/aiter/jit/module_moe_sorting.so $(location module_moe_sorting.so);
        cp external/aiter/aiter/jit/module_moe_asm.so $(location module_moe_asm.so);
        cp external/aiter/aiter/jit/module_moe.so $(location module_moe.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8_bpreshuffle.so $(location module_gemm_a8w8_bpreshuffle.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8_blockscale.so $(location module_gemm_a8w8_blockscale.so);
        cp external/aiter/aiter/jit/module_quant.so $(location module_quant.so);
        cp external/aiter/aiter/jit/module_pa.so $(location module_pa.so);
        cp external/composable_kernel_archive/libtile_example_fmha_fwd.so $(location libtile_example_fmha_fwd.so);
        cp external/composable_kernel_archive/libtile_rmsnorm2d_fwd.so $(location libtile_rmsnorm2d_fwd.so);
        cp external/aiter/aiter/jit/module_activation.so $(location module_activation.so);
    """,
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
        ":copy_hf3fs_libs",
    ] + kernel_so_deps()
)
