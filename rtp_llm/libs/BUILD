load("//bazel:arch_select.bzl", "kernel_so_deps", "copy_all_so")
package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_all_so()

genrule(
    name = "aiter_copy",
    srcs = ["@aiter//:cpp_libraries","@composable_kernel_archive//:ck_fmha_rmsnorm2d_libraries"],
    outs = [
        "libmodule_aiter_enum.so",
        "libmodule_custom_all_reduce.so",
        "libmodule_moe_sorting.so",
        "libmodule_moe_asm.so",
        "libmodule_moe.so",
        "libmodule_gemm_a8w8_bpreshuffle.so",
        "libmodule_gemm_a8w8_blockscale.so",
        "libmodule_quant.so",
        "libmodule_pa.so",
        "libtile_rmsnorm2d_fwd.so",
        "libtile_example_fmha_fwd.so",
        "libmodule_activation.so",
    ],
    cmd = """
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_aiter_enum.so $(location libmodule_aiter_enum.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_custom_all_reduce.so $(location libmodule_custom_all_reduce.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_moe_sorting.so $(location libmodule_moe_sorting.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_moe_asm.so $(location libmodule_moe_asm.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_moe.so $(location libmodule_moe.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_gemm_a8w8_bpreshuffle.so $(location libmodule_gemm_a8w8_bpreshuffle.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_gemm_a8w8_blockscale.so $(location libmodule_gemm_a8w8_blockscale.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_quant.so $(location libmodule_quant.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_pa.so $(location libmodule_pa.so);
        cp ./bazel-out/k8-opt/bin/external/composable_kernel_archive/libtile_example_fmha_fwd.so $(location libtile_example_fmha_fwd.so);
        cp ./bazel-out/k8-opt/bin/external/composable_kernel_archive/libtile_rmsnorm2d_fwd.so $(location libtile_rmsnorm2d_fwd.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_activation.so $(location libmodule_activation.so);
    """,
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
    ] + kernel_so_deps()
)
