load('@bazel_tools//tools/build_defs/repo:git.bzl', "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def git_deps():
    git_repository(
        name = "aiter_src",
        remote = "https://github.com/ROCm/aiter.git",
        commit = "329d07ba5d77f7d6b2a0557174288c5707f95e5f", # [Triton] DS a16w8 GEMM and fused reduce_rms_fp8_group_quant (#1328)
        recursive_init_submodules = True,
        patches = ["//3rdparty/aiter:aiter.patch", "//3rdparty/aiter:gemm_a8w8.patch"],
        patch_cmds = [
            "echo 'from aiter.jit.core import compile_ops, get_args_of_build, build_module, get_module' >> build_aiter_module.py",
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
            "echo '    build_aiter_module(\"module_aiter_enum\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_custom_all_reduce\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_quick_all_reduce\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_norm\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_rmsnorm\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_mha_fwd\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_fmha_v3_varlen_fwd\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_gemm_a8w8_blockscale\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_quant\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_smoothquant\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_moe_sorting\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_moe_asm\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_pa\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_attention_asm\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_activation\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_gemm_a8w8_bpreshuffle\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_gemm_a8w8\")' >> build_aiter_module.py",
            "echo '    build_aiter_module(\"module_moe_ck2stages\")' >> build_aiter_module.py",
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
        remote = "https://github.com/bazelbuild/rules_cc.git",
        commit = "1477dbab59b401daa94acedbeaefe79bf9112167",
    )

    git_repository(
        name = "rules_python",
        remote = "https://github.com/bazelbuild/rules_python.git",
        commit = "084b877c98b580839ceab2b071b02fc6768f3de6",
        patches = [
            "//patches/rules_python:0001-add-extra-data.patch",
            "//patches/rules_python:0002-remove-import-from-rules_cc.patch",
            "//patches/rules_python:0001-xx.patch",
        ],
    )

    new_git_repository(
        name = "cutlass_fa",
        remote = "https://github.com/NVIDIA/cutlass.git",
        commit = "bbe579a9e3beb6ea6626d9227ec32d0dae119a49",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass",
        remote = "https://github.com/NVIDIA/cutlass.git",
        commit = "80243e0b8c644f281e2beb0c20fe78cf7b267061",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass_h_moe",
        remote = "https://github.com/NVIDIA/cutlass.git",
        commit = "19b4c5e065e7e5bbc8082dfc7dbd792bdac850fc",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    new_git_repository(
        name = "cutlass3.6",
        remote = "https://github.com/NVIDIA/cutlass.git",
        commit = "cc3c29a81a140f7b97045718fb88eb0664c37bd7",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
        patches = ["//3rdparty/cutlass:0001-cuda12.4-compat.patch"],
    )

    new_git_repository(
        name = "cutlass4.0",
        remote = "https://github.com/NVIDIA/cutlass.git",
        commit = "dc4817921edda44a549197ff3a9dcf5df0636e7b",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    new_git_repository(
        name = "flashinfer_cpp",
        remote = "https://github.com/flashinfer-ai/flashinfer.git",
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
            "//3rdparty/flashinfer:0009-sp-sample.patch"
        ],
    )

    new_git_repository(
        name = "flashmla",
        remote = "https://github.com/deepseek-ai/FlashMLA.git",
        commit = "b31bfe72a83ea205467b3271a5845440a03ed7cb",
        build_file = str(Label("//3rdparty/flashmla:flashmla.BUILD")),
        patches = [
            "//3rdparty/flashmla:0001-add-interface.patch",
        ],
    )

    new_git_repository(
        name = "flash_attention",
        remote = "https://github.com/Dao-AILab/flash-attention.git",
        # v2.5.6
        commit = "6c9e60de566800538fedad2ad5e6b7b55ca7f0c5",
        patches = [
            "//patches/flash_attention:0001-fix-fix-arch-80-compile.patch",
            "//patches/flash_attention:0002-fix-remove-torch-aten-dep.patch",
            "//patches/flash_attention:0003-fix-fix-is-local-judge.patch",
        ],
        build_file = str(Label("//3rdparty/flash_attention:flash_attention.BUILD")),
    )

    git_repository(
        name = "com_google_googletest",
        remote = "https://github.com/google/googletest.git",
        commit = "f8d7d77c06936315286eb55f8de22cd23c188571",
        shallow_since = "1640057570 +0800",
    )

    http_archive(
        name = "com_github_nanopb_nanopb",
        sha256 = "8bbbb1e78d4ddb0a1919276924ab10d11b631df48b657d960e0c795a25515735",
        build_file = "@grpc//third_party:nanopb.BUILD",
        strip_prefix = "nanopb-f8ac463766281625ad710900479130c7fcb4d63b",
        urls = [
            "http://storage.googleapis.com/mirror.tensorflow.org/github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
            "http://github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
        ],
    )

    http_archive(
        name = "six_archive",
        build_file = clean_dep("//3rdparty/six:six.BUILD"),
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        urls = [
            "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            "http://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        ],
    )

    http_archive(
        name = "zlib_archive",
        build_file = clean_dep("//3rdparty/zlib:zlib.BUILD"),
        strip_prefix = "zlib-1.2.11",
        urls = [
            "https://www.zlib.net/fossils/zlib-1.2.11.tar.gz",
        ],
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    )

    git_repository(
        name = "com_google_absl",
        remote = "https://github.com/abseil/abseil-cpp.git",
        patch_cmds = [
            "sed -i -e 's/^#define ABSL_OPTION_USE_STD_STRING_VIEW 2/#define ABSL_OPTION_USE_STD_STRING_VIEW 0/' 'absl/base/options.h'",
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i absl/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(cpu|x86_32|x86_64|ppc|arm|aarch64|s390x)$@platforms//cpu:\\1$' -i -E absl/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(linux|osx|windows|android|freebsd|ios|os)$@platforms//os:\\1$' -E -i absl/time/internal/cctz/BUILD.bazel",
            "sed 's$@bazel_tools//platforms:(cpu|x86_32|x86_64|ppc|arm|aarch64|s390x)$@platforms//cpu:\\1$' -i -E absl/time/internal/cctz/BUILD.bazel",
        ],
        commit = "6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c",
        shallow_since = "1678195250 +0800",
    )

    native.local_repository(
        name = "com_google_protobuf",
        path = "3rdparty/protobuf",
    )

    new_git_repository(
        name = "rapidjson",
        remote = "https://github.com/Tencent/rapidjson.git",
        # tag = "v1.1.0",
        commit = "f54b0e47a08782a6131cc3d60f94d038fa6e0a51",
        patches = ["//3rdparty/rapidjson:0001-document_h.patch"],
        build_file = clean_dep("//3rdparty/rapidjson:rapidjson.BUILD"),
    )

    new_git_repository(
        name = "havenask",
        remote = "https://github.com/alibaba/havenask.git",
        commit = "3c973500afbd40933eb0a80cfdfb6592274377fb",
        shallow_since = "1704038400 +0800",
        patches=[
            "//patches/havenask:havenask.patch",
            "//patches/havenask:anet.patch",
            "//patches/havenask:0001-fix-PrometheusSink-need-header.patch"
        ],
        build_file = clean_dep("//3rdparty/kmonitor:kmonitor.BUILD"),
    )

    new_git_repository(
        name = "nacos_sdk_cpp",
        remote = "https://github.com/nacos-group/nacos-sdk-cpp.git",
        commit = "2b4104d2524776dff236a228ad2abff4676fb916",
        patches = [
            "//patches/nacos_sdk_cpp:nacos-compile.patch",
        ],
        build_file = clean_dep("//3rdparty/nacos_sdk_cpp:nacos_sdk_cpp.BUILD")
    )

    http_archive(
        name = "curl",
        build_file = clean_dep("//3rdparty/curl:curl.BUILD"),
        sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
        strip_prefix = "curl-7.60.0",
        urls = [
            "https://mirror.bazel.build/curl.haxx.se/download/curl-7.60.0.tar.gz",
            "https://curl.haxx.se/download/curl-7.60.0.tar.gz",
        ],
    )

    git_repository(
        name = "grpc",
        remote = "https://github.com/grpc/grpc.git",
        commit = "109c570727c3089fef655edcdd0dd02cc5958010",
        patches = ["//patches/grpc:0001-Rename-gettid-functions.patch"],
        shallow_since = "1518192000 +0800",
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
        sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
        strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
        urls = [
            "https://mirror.bazel.build/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
            "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
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

    new_git_repository(
        name = "flash-linear-attention",
        remote = "https://github.com/fla-org/flash-linear-attention.git",
        commit = "0d3e202a9c5a1a829ac3fe7c0a0c5fec0bf8f00b",
        patches = [
            "//3rdparty/flash_linear_attention:0001-modify-init.patch",
        ],
        build_file = str(Label("//3rdparty/flash_linear_attention:fla.BUILD")),
    )
