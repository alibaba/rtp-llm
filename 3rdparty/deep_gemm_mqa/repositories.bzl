"""Register the isolated FP32-output MQA backend with public or internal deps."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _native_mqa_repository(name, arch, sha256):
    # Older internal overlays may already register either repository.
    if native.existing_rule(name):
        return
    http_archive(
        name = name,
        urls = ["http://artlab.alibaba-inc.com/1/pypi/rtp_llm/deep_gemm/deep_gemm-2.8.0%2B122e18b.cu132-cp310-cp310-linux_{}.whl".format(arch)],
        sha256 = sha256,
        type = "zip",
        patches = [
            "@rtp_llm//patches/deep_gemm_mqa:mqa_only.patch",
            "@rtp_llm//patches/deep_gemm_mqa:isolate_globals.patch",
        ],
        patch_args = ["-p1"],
        patch_cmds = [
            "mv deep_gemm deep_gemm_mqa",
            "\"${PYTHON_BIN_PATH:-/opt/conda310/bin/python3}\" _k31_isolate.py deep_gemm_mqa",
            "rm _k31_isolate.py",
            "cp deep_gemm-2.8.0+122e18b.cu132.dist-info/LICENSE deep_gemm_mqa/LICENSE",
        ],
        build_file = "@rtp_llm//3rdparty/deep_gemm_mqa:deep_gemm_mqa.BUILD",
    )

def deep_gemm_mqa_deps():
    _native_mqa_repository(
        name = "k31_native_mqa",
        arch = "x86_64",
        sha256 = "5d5afc1d58bb24f7c8a460f74e8dff39a99d7a6ebae35152cdcfd5f393ead080",
    )
    _native_mqa_repository(
        name = "k31_native_mqa_arm",
        arch = "aarch64",
        sha256 = "31aa5238ca6c16334f28ac3364575ac606a6f79c424d8da52e41375057965a64",
    )
