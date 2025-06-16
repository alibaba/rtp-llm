load("//bazel:arch_select.bzl", "kernel_so_deps", "copy_all_so")
package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_all_so()

genrule(
    name = "aiter_copy",
    srcs = ["@aiter//:cpp_libraries"],
    outs = [
        "libasm_mla_decode_fwd_torch.so",
        "libmodule_gemm_a8w8_blockscale.so",
        "libmodule_quant.so"
    ],
    cmd = """
        cp ./bazel-out/k8-opt/bin/external/aiter/csrc/cpp_itfs/mla/libasm_mla_decode_fwd_torch.so $(location libasm_mla_decode_fwd_torch.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_gemm_a8w8_blockscale.so $(location libmodule_gemm_a8w8_blockscale.so);
        cp ./bazel-out/k8-opt/bin/external/aiter/aiter/jit/libmodule_quant.so $(location libmodule_quant.so);
    """,
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":libth_transformer_so",
    ] + kernel_so_deps()
)
