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
        "@//:using_aiter_src": [":aiter_src_copy",],
        "@//:using_rocm": [":aiter_copy"],
        "//conditions:default": [],
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
    srcs = ["@aiter_src//:cpp_libraries"],
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
        "libmodule_activation.so",
    ],
    cmd = """
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_aiter_enum.so $(location libmodule_aiter_enum.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_custom_all_reduce.so $(location libmodule_custom_all_reduce.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_moe_sorting.so $(location libmodule_moe_sorting.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_moe_asm.so $(location libmodule_moe_asm.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_moe.so $(location libmodule_moe.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_gemm_a8w8_bpreshuffle.so $(location libmodule_gemm_a8w8_bpreshuffle.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_gemm_a8w8_blockscale.so $(location libmodule_gemm_a8w8_blockscale.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_quant.so $(location libmodule_quant.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_pa.so $(location libmodule_pa.so);
        cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_activation.so $(location libmodule_activation.so);
    """,
    tags = ["rocm", "local"],
)

genrule(
    name = "aiter_copy",
    srcs = ["@aiter//:aiter_so"],
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
        cp external/aiter/aiter/jit/module_activation.so $(location module_activation.so);
    """,
    tags = ["rocm", "local"],
)

genrule(
    name = "ck_copy",
    srcs = ["@composable_kernel_archive//:ck_fmha_rmsnorm2d_libraries"],
    outs = [
        "libtile_rmsnorm2d_fwd.so",
        "libtile_example_fmha_fwd.so",
    ],
    cmd = """
        cp external/composable_kernel_archive/libtile_example_fmha_fwd.so $(location libtile_example_fmha_fwd.so);
        cp external/composable_kernel_archive/libtile_rmsnorm2d_fwd.so $(location libtile_rmsnorm2d_fwd.so);
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
        ":libth_transformer_config_so",
        ":libth_transformer_so"
    ] + kernel_so_deps()
)