load("//bazel:arch_select.bzl", "copy_all_so", "kernel_so_deps")
load("//bazel:defs.bzl", "copy_files")

package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_all_so()

# copy 3fs libs
copy_files(
    name = "copy_hf3fs_libs",
    srcs = select({
        "//:using_remote_kv_cache": ["//3rdparty/3fs:hf3fs_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

copy_files(
    name = "copy_remote_kv_cache_manager_client_libs",
    srcs = select({
        "//:using_remote_kv_cache": ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "copy_aiter",
    srcs = select({
        "@//:using_rocm_with_aiter_src": [
            ":aiter_src_copy",
            ":deep_ep_rocm_copy",
        ],
        "@//:using_aiter_src": [
            ":aiter_src_copy",
            ":deep_ep_rocm_copy",
        ],
        "@//:rocm_gfx950": [
            ":aiter_copy",
            ":aiter_copy_moe_ck2stages",
            ":deep_ep_rocm_copy",
        ],
        "@//:using_rocm": [
            ":aiter_copy",
            ":aiter_copy_moe_ck2stages",
            ":deep_ep_rocm_copy",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

copy_files(
    name = "copy_acclep_libs",
    srcs = select({
        "//rtp_llm/models_py/bindings/cuda/ops:use_accl_ep": ["//3rdparty/accl_ep:accl_ep_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# Separate genrule for libdeep_ep_rocm.so to avoid conflicts between aiter_src_copy and aiter_copy
genrule(
    name = "deep_ep_rocm_copy",
    srcs = select({
        "//rtp_llm/cpp/rocm_compat:enable_deep_ep": ["@deep_ep_rocm//:cpp_libraries"],
        "//conditions:default": [],
    }),
    outs = ["libdeep_ep_rocm.so"],
    cmd = select({
        "//rtp_llm/cpp/rocm_compat:enable_deep_ep": """
            # Copy libdeep_ep_rocm.so from @deep_ep_rocm//:cpp_libraries
            # $(SRCS) contains all files from srcs, filter for .so file
            for f in $(SRCS); do
                if [[ "$$f" == *.so ]]; then
                    cp "$$f" $(location libdeep_ep_rocm.so);
                    break;
                fi
            done
        """,
        "//conditions:default": """
            # Create a minimal valid shared library placeholder when deep_ep is not enabled
            echo "void dummy() {}" | gcc -shared -fPIC -x c - -o $(location libdeep_ep_rocm.so) || touch $(location libdeep_ep_rocm.so);
        """,
    }),
    tags = [
        "local",
        "rocm",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "aiter_src_copy",
    srcs = ["@aiter_src//:cpp_libraries"],
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
        "libmodule_deepgemm.so",
        "libmodule_attention_asm.so",
        "libmodule_mha_fwd.so",
        "libmodule_mha_batch_prefill.so",
        "libmodule_fmha_v3_varlen_fwd.so",
        "libmodule_norm.so",
        "libmodule_rmsnorm.so",
        "libmodule_gemm_a8w8.so",
        "libmodule_moe_ck2stages.so",
        "libmodule_gated_rmsnorm_quant.so",
        "libmodule_gemm_a8w8_blockscale_bpreshuffle.so",
        "libmodule_moe_fmoe_asm.so",
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
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_mha_batch_prefill.so $(location libmodule_mha_batch_prefill.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_fmha_v3_varlen_fwd.so $(location libmodule_fmha_v3_varlen_fwd.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_norm.so $(location libmodule_norm.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_rmsnorm.so $(location libmodule_rmsnorm.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_gemm_a8w8.so $(location libmodule_gemm_a8w8.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_moe_ck2stages.so $(location libmodule_moe_ck2stages.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_deepgemm.so $(location libmodule_deepgemm.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_gated_rmsnorm_quant.so $(location libmodule_gated_rmsnorm_quant.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_gemm_a8w8_blockscale_bpreshuffle.so $(location libmodule_gemm_a8w8_blockscale_bpreshuffle.so);
cp ./bazel-out/k8-opt/bin/external/aiter_src/aiter/jit/libmodule_moe_fmoe_asm.so $(location libmodule_moe_fmoe_asm.so);
""",
    tags = [
        "local",
        "rocm",
    ],
)

genrule(
    name = "aiter_copy",
    srcs = ["@aiter//:aiter_so"],
    outs = [
        "module_aiter_core.so",
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
        "module_mha_batch_prefill.so",
        "module_fmha_v3_varlen_fwd.so",
        "module_norm.so",
        "module_rmsnorm.so",
        "module_gemm_a8w8.so",
        "module_deepgemm.so",
        "module_gated_rmsnorm_quant.so",
        "module_gemm_a8w8_blockscale_bpreshuffle.so",
        "module_moe_fmoe_asm.so",
    ],
    cmd = """
        cp external/aiter/aiter/jit/module_aiter_core.so $(location module_aiter_core.so);
        cp external/aiter/aiter/jit/module_custom_all_reduce.so $(location module_custom_all_reduce.so);
        cp external/aiter/aiter/jit/module_quick_all_reduce.so $(location module_quick_all_reduce.so);
        cp external/aiter/aiter/jit/module_moe_sorting.so $(location module_moe_sorting.so);
        cp external/aiter/aiter/jit/module_moe_asm.so $(location module_moe_asm.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8_bpreshuffle.so $(location module_gemm_a8w8_bpreshuffle.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8_blockscale.so $(location module_gemm_a8w8_blockscale.so);
        cp external/aiter/aiter/jit/module_quant.so $(location module_quant.so);
        cp external/aiter/aiter/jit/module_smoothquant.so $(location module_smoothquant.so);
        if [[ -f external/aiter/aiter/jit/module_pa.so ]]; then
            cp external/aiter/aiter/jit/module_pa.so $(location module_pa.so);
        elif [[ "$${GPU_ARCHS:-}" == *gfx950* ]]; then
            # The gfx950 wheel uses module_attention_asm and has no module_pa.
            # Keep the declared output valid without weakening other ROCm builds.
            echo "void rtp_llm_unused_module_pa() {}" | gcc -shared -fPIC -x c - -o $(location module_pa.so);
        else
            echo "AITER wheel is missing required module_pa.so for GPU_ARCHS=$${GPU_ARCHS:-unknown}" >&2;
            exit 1;
        fi;
        cp external/aiter/aiter/jit/module_activation.so $(location module_activation.so);
        cp external/aiter/aiter/jit/module_attention_asm.so $(location module_attention_asm.so);
        cp external/aiter/aiter/jit/module_mha_fwd.so $(location module_mha_fwd.so);
        cp external/aiter/aiter/jit/module_mha_batch_prefill.so $(location module_mha_batch_prefill.so);
        cp external/aiter/aiter/jit/module_fmha_v3_varlen_fwd.so $(location module_fmha_v3_varlen_fwd.so);
        cp external/aiter/aiter/jit/module_norm.so $(location module_norm.so);
        cp external/aiter/aiter/jit/module_rmsnorm.so $(location module_rmsnorm.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8.so $(location module_gemm_a8w8.so);
        cp external/aiter/aiter/jit/module_deepgemm.so $(location module_deepgemm.so);
        cp external/aiter/aiter/jit/module_gated_rmsnorm_quant.so $(location module_gated_rmsnorm_quant.so);
        cp external/aiter/aiter/jit/module_gemm_a8w8_blockscale_bpreshuffle.so $(location module_gemm_a8w8_blockscale_bpreshuffle.so);
        cp external/aiter/aiter/jit/module_moe_fmoe_asm.so $(location module_moe_fmoe_asm.so);
""",
    tags = [
        "local",
        "rocm",
    ],
)

genrule(
    name = "aiter_copy_moe_ck2stages",
    srcs = ["@aiter//:aiter_so"],
    outs = ["module_moe_ck2stages.so"],
    cmd = """
        if [[ -f external/aiter/aiter/jit/module_moe_ck2stages.so ]]; then
            cp external/aiter/aiter/jit/module_moe_ck2stages.so $(location module_moe_ck2stages.so);
        elif [[ "$${GPU_ARCHS:-}" == *gfx950* ]]; then
            # gfx950 uses other MoE implementations, but some RTP-LLM shared
            # libraries retain a DT_NEEDED entry for this optional AITER DSO.
            echo "void rtp_llm_unused_module_moe_ck2stages() {}" | gcc -shared -fPIC -x c - -o $(location module_moe_ck2stages.so);
        else
            echo "AITER wheel is missing required module_moe_ck2stages.so for GPU_ARCHS=$${GPU_ARCHS:-unknown}" >&2;
            exit 1;
        fi
    """,
    tags = [
        "local",
        "rocm",
    ],
)

filegroup(
    name = "frontend_libs",
    srcs = [],
    data = [
        ":libth_transformer_config_so",
    ],
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":copy_acclep_libs",
        ":copy_aiter",
        ":copy_hf3fs_libs",
        ":copy_remote_kv_cache_manager_client_libs",
    ] + kernel_so_deps(),
)

filegroup(
    name = "whl_package_libs",
    srcs = [],
    data = [
        ":librtp_compute_ops_so",
        ":libth_transformer_config_so",
        ":libth_transformer_so",
    ],
)
