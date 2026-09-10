load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def git_deps():

    git_repository(
        name = "aiter_src",
        remote = "https://github.com/ROCm/aiter.git",
	commit = "a75b522b314f0c5af96acc3b11efe580973586f0", # fix kernel repeat loading error (#1759)
        recursive_init_submodules = True,
        patches = ["//3rdparty/aiter:aiter.patch",
                   "//3rdparty/aiter:gemm_a8w8.patch"],
        patch_cmds = [
            "echo 'from aiter.jit.core import compile_ops, get_args_of_build, build_module, get_module' >> build_aiter_module.py",
            "echo 'import multiprocessing' >> build_aiter_module.py",
            "echo 'from typing import Dict' >> build_aiter_module.py",
            "echo 'import os' >> build_aiter_module.py",
            "echo '' >> build_aiter_module.py",
            "echo 'def build_aiter_module(md_name: str, custom_build_args: Dict = {}):' >> build_aiter_module.py",
            "echo '    if os.path.exists(f\"aiter/jit/{md_name}.so\"):' >> build_aiter_module.py",
            "echo '        return' >> build_aiter_module.py",
            "echo '' >> build_aiter_module.py",
            "echo '    d_args = get_args_of_build(md_name)' >> build_aiter_module.py",
            "echo '    d_args.update(custom_build_args)' >> build_aiter_module.py",
            "echo '' >> build_aiter_module.py",
            "echo '    md_name = custom_build_args.get(\"md_name\", md_name)' >> build_aiter_module.py",
            "echo '' >> build_aiter_module.py",

            "echo '    srcs = d_args[\"srcs\"]' >> build_aiter_module.py",
            "echo '    flags_extra_cc = d_args[\"flags_extra_cc\"]' >> build_aiter_module.py",
            "echo '    flags_extra_hip = d_args[\"flags_extra_hip\"]' >> build_aiter_module.py",
            "echo '    blob_gen_cmd = d_args[\"blob_gen_cmd\"]' >> build_aiter_module.py",
            "echo '    extra_include = d_args[\"extra_include\"]' >> build_aiter_module.py",
            "echo '    extra_ldflags = d_args[\"extra_ldflags\"]' >> build_aiter_module.py",
            "echo '    verbose = d_args[\"verbose\"]' >> build_aiter_module.py",
            "echo '    is_python_module = d_args[\"is_python_module\"]' >> build_aiter_module.py",
            "echo '    is_standalone = d_args[\"is_standalone\"]' >> build_aiter_module.py",
            "echo '    torch_exclude = d_args[\"torch_exclude\"]' >> build_aiter_module.py",
            "echo '    module = build_module(' >> build_aiter_module.py",
            "echo '                         md_name,' >> build_aiter_module.py",
            "echo '                         srcs,' >> build_aiter_module.py",
            "echo '                         flags_extra_cc,' >> build_aiter_module.py",
            "echo '                         flags_extra_hip,' >> build_aiter_module.py",
            "echo '                         blob_gen_cmd,' >> build_aiter_module.py",
            "echo '                         extra_include,' >> build_aiter_module.py",
            "echo '                         extra_ldflags,' >> build_aiter_module.py",
            "echo '                         verbose,' >> build_aiter_module.py",
            "echo '                         is_python_module,' >> build_aiter_module.py",
            "echo '                         is_standalone,' >> build_aiter_module.py",
            "echo '                         torch_exclude,' >> build_aiter_module.py",
            "echo '    )' >> build_aiter_module.py",
            "echo 'if __name__ == \"__main__\":' >> build_aiter_module.py",
            "echo '    # prebuild module_aiter_enum, which is needed by other modules' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_aiter_enum\")' >> build_aiter_module.py",
            "echo '    module_names = []' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_custom_all_reduce\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_norm\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_rmsnorm\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_mha_fwd\")' >> build_aiter_module.py",
	    "echo '    module_names.append(\"module_mha_batch_prefill\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_fmha_v3_varlen_fwd\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_gemm_a8w8_blockscale\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_quant\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_smoothquant\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_moe_sorting\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_moe_asm\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_pa\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_attention_asm\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_activation\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_gemm_a8w8_bpreshuffle\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_gemm_a8w8\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_moe_ck2stages\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_deepgemm\")' >> build_aiter_module.py",
            "echo '    module_names.append(\"module_quick_all_reduce\")' >> build_aiter_module.py",
            "echo '    with multiprocessing.Pool(processes = 64) as pool:' >> build_aiter_module.py",
            "echo '        pool.map(build_aiter_module, module_names)' >> build_aiter_module.py",
            "echo 'echo \"building mla kernel\"' >> build_mla_kernel.sh",
            "echo 'so_file=\"./csrc/cpp_itfs/mla/asm_mla_decode_fwd_torch_lib.so\"' >> build_mla_kernel.sh",
            "echo 'if [ -f $so_file ]; then' >> build_mla_kernel.sh",
            "echo '    exit 0' >> build_mla_kernel.sh",
            "echo 'else' >> build_mla_kernel.sh",
            "echo '    export PYTHONPATH=`pwd`:$PYTHONPATH' >> build_mla_kernel.sh",
            "echo '    /opt/conda310/bin/python aiter/aot/asm_mla_decode_fwd.py' >> build_mla_kernel.sh",
            "echo '    cd ./csrc/cpp_itfs/mla' >> build_mla_kernel.sh",
            "echo '    make asm_mla_decode_fwd_torch_lib.so' >> build_mla_kernel.sh",
            "echo 'fi' >> build_mla_kernel.sh",

        ],
        build_file = "//3rdparty/aiter:BUILD",
    )

    git_repository(
        name = "rules_cc",
        remote = "git@gitlab.alibaba-inc.com:search_external/rules_cc.git",
        commit = "ab9d9620e247cb3f759a007c0f78f6673a7b8cf8",
    )

    git_repository(
        name = "rules_python",
        remote = "git@gitlab.alibaba-inc.com:search_external/rules_python.git",
        # https://github.com/bazelbuild/rules_python/releases/tag/0.34.0
        commit = "084b877c98b580839ceab2b071b02fc6768f3de6",
        patches = [
            "//patches/rules_python:0001-add-extra-data.patch",
            "//patches/rules_python:0002-remove-import-from-rules_cc.patch",
            "//patches/rules_python:0001-xx.patch",
        ],
    )

    new_git_repository(
        name = "cutlass_ppu",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/cutlass.git",
        commit = "7d9e2007a22d43e8a0f02e84ed40a3954ba58745",
        patch_cmds = [" bash tools/replace_cudart.sh "],
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/cutlass_ppu:cutlass.BUILD")),
        patches = [
            "//internal_source/RTP_LLM-PPU/3rdparty/cutlass_ppu:0001-DeepGeem_1v51.patch"
        ],
    )

    new_git_repository(
        name = "cutlass_ppu_acext",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/cutlass.git",
        commit = "7d9e2007a22d43e8a0f02e84ed40a3954ba58745",
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/cutlass_ppu:cutlass.BUILD")),
        patches = [
            "//internal_source/RTP_LLM-PPU/3rdparty/cutlass_ppu:0001-DeepGeem_1v51.patch"
        ],
    )

    new_git_repository(
        name = "cutlass3_ppu_flashinfer",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/cutlass3.git",
        commit = "3b5c01a7f9ba2c6f737123913a8d8dc01eeef3dd",
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/cutlass_ppu:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass3_ppu_flashmla",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/cutlass3.git",
        commit = "84578e73737387d8c3b42254541c4a46416e3a90",
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/cutlass_ppu:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass3_ppu",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/cutlass3.git",
        commit = "3e79fb2c6dc5c1f868d18b6d34ae2a655a9afae1",
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/cutlass_ppu:cutlass.BUILD")),
    )

    new_git_repository(
        name = "flash_attention",
        remote = "git@gitlab.alibaba-inc.com:foundation_models/flash-attention.git",
        # v2.5.6
        commit = "6c9e60de566800538fedad2ad5e6b7b55ca7f0c5",
        patches = [
            "//patches/flash_attention:0001-fix-fix-arch-80-compile.patch",
            "//patches/flash_attention:0002-fix-remove-torch-aten-dep.patch",
            "//patches/flash_attention:0003-fix-fix-is-local-judge.patch",
        ],
        build_file = str(Label("//3rdparty/flash_attention:flash_attention.BUILD")),
    )

    new_git_repository(
        name = "flash_attention_ppu",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/flash-attention.git",
        commit = "7548bd228fff8ec5d406cf28cf20d594f7c7a699",
        patches = [
            "//internal_source/RTP_LLM-PPU/3rdparty/flash_attention_ppu:0001-fix-fix-is-local-judge.patch",
            #"//internal_source/RTP_LLM-PPU/3rdparty/flash_attention_ppu:0002-fa-perf.patch",
        ],
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/flash_attention_ppu:flash_attention.BUILD")),
    )

    new_git_repository(
        name = "acext",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/acext.git",
        commit = "8f2f3a4d989ce5f8a46f5b5b35726bf54f8c2652",
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/acext:acext.BUILD")),
    )

    new_git_repository(
        name = "cutlass",
        remote = "git@gitlab.alibaba-inc.com:search_external/cutlass.git",
        commit = "80243e0b8c644f281e2beb0c20fe78cf7b267061",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    # cutlass_cu13: CUTLASS v3.8 "3.8 v2" — same CUDA 13 readiness as
    # cutlass3.6_cu13; used by flashmla + trt_fused_multihead_attention when
    # the build is configured for cuda13_arm.
    new_git_repository(
        name = "cutlass_cu13",
        remote = "git@gitlab.alibaba-inc.com:search_external/cutlass.git",
        commit = "b84e9802d84b16bcb4e92338fcf0a04785df9236",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass_h_moe",
        remote = "git@gitlab.alibaba-inc.com:search_external/cutlass.git",
        commit = "19b4c5e065e7e5bbc8082dfc7dbd792bdac850fc",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass_fa",
        remote = "git@gitlab.alibaba-inc.com:search_external/cutlass.git",
        commit = "bbe579a9e3beb6ea6626d9227ec32d0dae119a49",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass3.6",
        remote = "git@gitlab.alibaba-inc.com:search_external/cutlass.git",
        commit = "cc3c29a81a140f7b97045718fb88eb0664c37bd7",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
        patches = ["//3rdparty/cutlass:0001-cuda12.4-compat.patch"],
    )

    # cutlass3.6_cu13: CUTLASS v3.8 "3.8 v2" (b84e9802) — natively gates
    # cuTensorMapEncodeTiled/Im2col on CUDA version so CUDA 13 can use the
    # by-version driver-entry-point API (PFN_*_v12000).  Sits alongside the
    # CUDA-12-targeted cutlass3.6 above; selected only by the cuda13_arm config.
    new_git_repository(
        name = "cutlass3.6_cu13",
        remote = "git@gitlab.alibaba-inc.com:search_external/cutlass.git",
        commit = "b84e9802d84b16bcb4e92338fcf0a04785df9236",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass4.0",
        remote = "git@gitlab.alibaba-inc.com:search_external/cutlass.git",
        commit = "dc4817921edda44a549197ff3a9dcf5df0636e7b",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    native.new_local_repository(
        name = "nvshmem",
        path = "/",
        build_file=str(Label("//3rdparty/nvshmem:nvshmem.BUILD")),
    )

    native.new_local_repository(
        name = "nvshmem_rocm",
        path = "/",
        build_file=str(Label("//3rdparty/nvshmem:nvshmem_rocm.BUILD")),
    )

    native.new_local_repository(
        name = "nvshmem_ppu",
        path = "/usr/local/PPU_SDK/sailSHMEM/",
        build_file=str(Label("//internal_source/RTP_LLM-PPU/3rdparty/nvshmem_ppu:nvshmem_ppu.BUILD")),
    )

    new_git_repository(
        name = "deep_ep_rocm",
        remote = "git@gitlab.alibaba-inc.com:meixue.lyh/deepep_rocm.git",
        commit = "366afecd8b9b202d5e1a53ba33f23920aaf07268",
        build_file = str(Label("//3rdparty/deep_ep:deep_ep_rocm.BUILD")),
    )

    new_git_repository(
        name = "deep_ep",
        remote = "git@gitlab.alibaba-inc.com:foundation_models/DeepEP.git",
        commit = "b86c6640bd94d865d13b117a40b4d3969f7b3b33",
        build_file = str(Label("//3rdparty/deep_ep:deep_ep.BUILD")),
    )

    new_git_repository(
        name = "deep_ep_ppu",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/DeepEP.git",
        commit = "9c7ffc9c69891525f0334cee54fd56588c76e9a0",
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/deep_ep_ppu:deep_ep_ppu.BUILD")),
        patches = [
            "//internal_source/RTP_LLM-PPU/3rdparty/deep_ep_ppu:0001-add-interface.patch"
        ]
    )

    new_git_repository(
        name = "flashmla",
        remote = "git@gitlab.alibaba-inc.com:foundation_models/FlashMLA.git",
        commit = "b31bfe72a83ea205467b3271a5845440a03ed7cb",
        build_file = str(Label("//3rdparty/flashmla:flashmla.BUILD")),
        patches = [
            "//3rdparty/flashmla:0001-add-interface.patch",
        ],
    )

    new_git_repository(
        name = "flashmla_ppu",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/FlashMLA.git",
        commit = "ad51bd3877ff7666da2402c695f8008d6ebf38f6",
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/flashmla:flashmla.BUILD")),
        patches = [
            "//internal_source/RTP_LLM-PPU/3rdparty/flashmla:0001-add-interface.patch",
        ],
    )

    new_git_repository(
        name = "flashinfer_cpp",
        remote = "git@gitlab.alibaba-inc.com:foundation_models/flashinfer.git",
        commit = "1c88d650eeec97be3a4dcebe4a9912d7785bc250",
        build_file = str(Label("//3rdparty/flashinfer:flashinfer.BUILD")),
        patches = [
            "//3rdparty/flashinfer:0001-fix-compile.patch",
            "//3rdparty/flashinfer:0002-dispatch-group-size.patch",
            "//3rdparty/flashinfer:0003-tanh-compatibility.patch",
            "//3rdparty/flashinfer:0005-update-add-mla-attn-test-impl-mla-write-kvcache.patch",
            "//3rdparty/flashinfer:0006-add-mla-dispatch-inc.patch",
            "//3rdparty/flashinfer:0007-fix-nan.patch",
            "//3rdparty/flashinfer:0008-enable-pdl.patch",
            "//3rdparty/flashinfer:0009-sp-sample.patch",
            "//3rdparty/flashinfer:0010-silu-mul-vec-size.patch",
        ],
    )

    # flashinfer_cpp_cu13: same 1c88d650 commit as flashinfer_cpp, with an
    # additional narrow patch (0011-cuda13-cub-compat) that backports the
    # upstream `#if CUDA_VERSION >= 12090 → cuda::maximum<>` guard into
    # sampling.cuh so CUDA 13's CUB (which dropped cub::Max/cub::Min) compiles.
    # A full upstream upgrade (9a79b78 / da1c3d2) would pull in flashinfer's
    # JIT refactor and require rewriting the aot_build_utils-driven bazel
    # BUILD file — out of scope here; this variant keeps the BUILD unchanged
    # and is selected only by the cuda13_arm config.
    new_git_repository(
        name = "flashinfer_cpp_cu13",
        remote = "git@gitlab.alibaba-inc.com:foundation_models/flashinfer.git",
        commit = "1c88d650eeec97be3a4dcebe4a9912d7785bc250",
        build_file = str(Label("//3rdparty/flashinfer:flashinfer_cu13.BUILD")),
        patches = [
            "//3rdparty/flashinfer:0001-fix-compile.patch",
            "//3rdparty/flashinfer:0002-dispatch-group-size.patch",
            "//3rdparty/flashinfer:0003-tanh-compatibility.patch",
            "//3rdparty/flashinfer:0005-update-add-mla-attn-test-impl-mla-write-kvcache.patch",
            "//3rdparty/flashinfer:0006-add-mla-dispatch-inc.patch",
            "//3rdparty/flashinfer:0007-fix-nan.patch",
            "//3rdparty/flashinfer:0008-enable-pdl.patch",
            "//3rdparty/flashinfer:0009-sp-sample.patch",
            "//3rdparty/flashinfer:0010-silu-mul-vec-size.patch",
            "//3rdparty/flashinfer:0011-cuda13-cub-compat.patch",
            "//3rdparty/flashinfer:0012-pymoduledef-missing-fields.patch",
            "//3rdparty/flashinfer:0013-cuda13-kernel-visibility-scheduler.patch",
            "//3rdparty/flashinfer:0014-cuda13-kernel-visibility-decode.patch",
            "//3rdparty/flashinfer:0015-cuda13-occupancy-skip.patch",
        ],
    )

    new_git_repository(
        name = "flashinfer_ppu",
        remote = "git@gitlab.alibaba-inc.com:ppu_open_source/flashinfer.git",
        commit = "9475b90cf5a03c6e698697c05578b8aa56e9911b",
        build_file = str(Label("//internal_source/RTP_LLM-PPU/3rdparty/flashinfer:flashinfer.BUILD")),
        patches = [
            "//3rdparty/flashinfer:0002-dispatch-group-size.patch",
            "//3rdparty/flashinfer:0005-update-add-mla-attn-test-impl-mla-write-kvcache.patch",
            "//3rdparty/flashinfer:0006-add-mla-dispatch-inc.patch",
	        "//internal_source/RTP_LLM-PPU/3rdparty/flashinfer:0007-fix-nan.patch",
            "//3rdparty/flashinfer:0009-sp-sample.patch",
            #"//internal_source/RTP_LLM-PPU/3rdparty/flashinfer:0008-remove-sm-limit.patch",
        ],
    )

    git_repository(
        name = "com_google_googletest",
        remote = "git@gitlab.alibaba-inc.com:search_external/gtest.git",
        commit = "1a9f2cf450187ff4e52ad8fc6dae4aaac6924c7b",
    )

    http_archive(
        name = "com_github_nanopb_nanopb",
        sha256 = "8bbbb1e78d4ddb0a1919276924ab10d11b631df48b657d960e0c795a25515735",
        build_file = "@grpc//third_party:nanopb.BUILD",
        strip_prefix = "nanopb-f8ac463766281625ad710900479130c7fcb4d63b",
        urls = [
            "http://pythonrun.oss-cn-zhangjiakou.aliyuncs.com/mirror/storage.googleapis.com/mirror.tensorflow.org/github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
            "http://pythonrun.oss-cn-zhangjiakou.aliyuncs.com/mirror/github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
        ],
    )

    http_archive(
        name = "six_archive",
        build_file = clean_dep("//3rdparty/six:six.BUILD"),
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        urls = [
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        ],
    )

    http_archive(
        name = "zlib_archive",
        build_file = clean_dep("//3rdparty/zlib:zlib.BUILD"),
        strip_prefix = "zlib-1.2.11",
        urls = [
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/aios/third_party_archives/zlib-1.2.11.tar.gz",
        ],
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    )

    git_repository(
        name = "com_google_absl",
        remote = "git@gitlab.alibaba-inc.com:search_external/abseil-cpp.git",
        patch_cmds = [
            "sed -i -e 's/^#define ABSL_OPTION_USE_STD_STRING_VIEW 2/#define ABSL_OPTION_USE_STD_STRING_VIEW 0/' 'absl/base/options.h'",
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i absl/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(cpu|x86_32|x86_64|ppc|arm|aarch64|s390x)$@platforms//cpu:\\1$' -i -E absl/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i absl/time/internal/cctz/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(cpu|x86_32|x86_64|ppc|arm|aarch64|s390x)$@platforms//cpu:\\1$' -i -E absl/time/internal/cctz/BUILD.bazel",
        ],
        commit = "6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c",
    )

    native.local_repository(
        name = "com_google_protobuf",
        path = "3rdparty/protobuf",
    )

    native.new_local_repository(
        name = "alibaba_rdma",
        path = "internal_source/rdma",
        build_file=clean_dep("//internal_source:rdma.BUILD")
    )

    git_repository(
        name = "grpc",
        remote = "git@gitlab.alibaba-inc.com:search_external/grpc.git",
        commit = "109c570727c3089fef655edcdd0dd02cc5958010",
        patches = ["//patches/grpc:0001-Rename-gettid-functions.patch"],
    )

    new_git_repository(
        name = "rapidjson",
        remote = "git@gitlab.alibaba-inc.com:isearch/rapidjson-mirror.git",
        # tag = "v1.1.0",
        commit = "f54b0e47a08782a6131cc3d60f94d038fa6e0a51",
	patches = ["//3rdparty/rapidjson:0001-document_h.patch"],
        build_file = clean_dep("//3rdparty/rapidjson:rapidjson.BUILD"),
    )

    new_git_repository(
        name = "havenask",
        remote = "git@gitlab.alibaba-inc.com:xijie.xuxj/havenask.git",
        commit = "f7317cb70a2e04c62d3b6a5f7f3c6ae75ab2149c",
        patches = [
            "//patches/havenask:havenask.patch",
            "//patches/havenask:anet.patch",
            "//patches/havenask:cm2.patch",
            "//patches/havenask:0001-fix-PrometheusSink-need-header.patch",
        ],
        build_file = clean_dep("//3rdparty/kmonitor:kmonitor.BUILD"),
    )

    new_git_repository(
        name = "nacos_sdk_cpp",
        remote = "git@gitlab.alibaba-inc.com:search_external/nacos-sdk-cpp.git",
        commit = "2b4104d2524776dff236a228ad2abff4676fb916",
        patches = [
            "//patches/nacos_sdk_cpp:nacos-compile.patch",
        ],
        build_file = clean_dep("//3rdparty/nacos_sdk_cpp:nacos_sdk_cpp.BUILD")
    )

    new_git_repository(
        name = "vipserver",
        remote = "git@gitlab.alibaba-inc.com:search_external/vipserver4c.git",
        build_file = clean_dep("//3rdparty/vipserver:vipserver.BUILD"),
        #based on tag "t-midware-vipserver-c-client_A_1_0_12_4_1126043_20170330"
        commit = "66f782fdb8c56a8ad72e2629dd30fc52f27becd6",
        shallow_since = "1667292833 +0800",
    )

    http_archive(
        name = "curl",
        build_file = clean_dep("//3rdparty/curl:curl.BUILD"),
        sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
        strip_prefix = "curl-7.60.0",
        urls = [
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/mirror.bazel.build/curl.haxx.se/download/curl-7.60.0.tar.gz",
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/curl.haxx.se/download/curl-7.60.0.tar.gz",
        ],
    )

    git_repository(
        name = "KleidiAI",
        remote = "https://git.gitlab.arm.com/kleidi/kleidiai.git",
        commit = "2d160cf675d6df7068a17da07ec6218fd9478541",
        patch_args = ["-p1"],
        patches = ["//patches/kai:0001-add-a8w4-fp16-support.patch"],
    )

    http_archive(
        name = "boringssl",
        # build_file = clean_dep("//3rdparty/boringssl:boringssl.BUILD"),
        sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
        strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
        urls = [
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/mirror.bazel.build/boringssl-7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
            "http://search-cicd.oss-cn-hangzhou-zmf.aliyuncs.com/odps_tensorflow/other/raw/master/boringssl-7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        ],
    )

    http_file(
        name = "krb5-devel",
        urls = ["http://mirrors.aliyun.com/centos/7/os/x86_64/Packages/krb5-devel-1.15.1-50.el7.x86_64.rpm"],
        sha256 = "75069ac38fed957b70ea1de5e2824e6a77468e9745a3a828d47a02bab727ba11",
    )

    http_file(
        name = "libcom_err-devel",
        urls = ["https://mirrors.aliyun.com/centos/7/os/x86_64/Packages/libcom_err-devel-1.42.9-19.el7.x86_64.rpm"],
        sha256 = "3a14db2d86490211494bb142139121da838160fb7ba28d46cd01568b0173969c",
    )

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@grpc//:grpc_cpp_plugin",
    )

    native.bind(
        name = "grpc_python_plugin",
        actual = "@grpc//:grpc_python_plugin",
    )

    # Needed by gRPC
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    # Needed by gRPC
    native.bind(
        name = "nanopb",
        actual = "@com_github_nanopb_nanopb//:nanopb",
    )

    # gRPC expects //external:protobuf_clib and //external:protobuf_compiler
    # to point to Protobuf's compiler library.
    native.bind(
        name = "protobuf_clib",
        actual = "@com_google_protobuf//:protoc_lib",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf_headers",
        actual = "@com_google_protobuf//:protobuf_headers",
    )

    # # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@grpc//:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@grpc//:grpc_python_plugin",
    )

    # # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

    # Needed by gRPC
    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )
