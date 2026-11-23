"""Pinned upstream sources for the isolated Blackwell extension ABIs."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@rtp_blackwell_config//:pins.bzl", "CUDA_EXTENSION", "DEEP_GEMM_WHEEL", "MHC_SOURCE")

BLACKWELL_SOURCES = {
    "cutlass_mhc": struct(
        repository = "NVIDIA/cutlass",
        commit = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
        sha256 = "2e5890306557bd87136e1d6914e4ee42536050b9116433848175520fd15991f2",
        srcs = ["include/cutlass/**", "include/cute/**"],
    ),
    "cutlass_mla": struct(
        repository = "NVIDIA/cutlass",
        commit = "4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b",
        sha256 = "50a017597cb76ab5dfe15a2419dd445010aef731b5ff9354e8da6543de204b23",
        srcs = ["include/**", "tools/util/include/**"],
    ),
    "deep_gemm": MHC_SOURCE,
    # This revision exposes the legacy FP8 indexer ABI. Keep its cubins separate.
    "deep_gemm_mqa": struct(
        repository = "deepseek-ai/DeepGEMM",
        commit = "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
        sha256 = "8a6c67b6f74d4a3a242c4dad7714d147797b2ecc5d55c52925bbdf6155207d2c",
        srcs = ["csrc/**", "deep_gemm/include/deep_gemm/**"],
    ),
    "deep_jit": struct(
        repository = "deepseek-ai/DeepJIT",
        commit = "2efdab421e1cfb17fe8bc20e11ca72aa6d0e6c43",
        sha256 = "e90e7a7d7643fee5514cf2efa73d96b1d104f87993266ea74d1205859e70c0c0",
        srcs = ["include/deep_jit/**"],
    ),
    "flash_mla": struct(
        repository = "deepseek-ai/FlashMLA",
        commit = "0f11e178da1f67becb087a324b400854931ec3e3",
        sha256 = "e94fe4a1844d8f6a551d78ce390de51c95e2b4e3e2d5d690c3e49dde9d6baa10",
        srcs = ["csrc/api/common.h", "csrc/cuda_kernels/**", "csrc/3rdparty/kerutils/include/**", "csrc/*.h"],
    ),
    "fmt": struct(
        repository = "fmtlib/fmt",
        commit = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28",
        sha256 = "c314292789d28c3c3b420e75a7b2d1706f685f7fb63289128d46aeaea2c6be71",
        srcs = ["include/fmt/**"],
    ),
}

# Git mirror of the Sourceware elfutils-0.191 tag, available to build workers.
ELFUTILS = struct(
    repository = "sailfishos-mirror/elfutils",
    commit = "18a015c0b0787ba5acb39801ab7c17dac50f584d",
    sha256 = "cf08bb76105161967f83225a20cee3e82c81839e07065ff10450cee8a2327aaf",
)

_ELFUTILS_BUILD = '''exports_files([
    "libdwfl/libdwfl.h",
    "libdw/libdw.h",
    "libelf/gelf.h",
    "libelf/libelf.h",
], visibility = ["//visibility:public"])'''

_BUILD = """
package(default_visibility = ["//visibility:public"])

filegroup(name = "sources", srcs = glob({srcs}, allow_empty = False))
filegroup(name = "license", srcs = glob(["LICENSE*"], allow_empty = False))
"""

def _wheel_lock_impl(ctx):
    ctx.file("BUILD.bazel", 'exports_files(["requirements.txt"])')
    ctx.file("requirements.txt", "deep-gemm @ {} --hash=sha256:{}\n".format(ctx.attr.url, ctx.attr.sha256))

_wheel_lock = repository_rule(
    implementation = _wheel_lock_impl,
    attrs = {"url": attr.string(mandatory = True), "sha256": attr.string(mandatory = True)},
)

def _deep_gemm_wheel_sources_impl(ctx):
    # The published wheel includes the original csrc and JIT headers.
    ctx.download_and_extract(ctx.attr.url, output = "wheel", sha256 = ctx.attr.sha256, type = "zip")
    dist_info = [p for p in ctx.path("wheel").readdir() if p.basename.endswith(".dist-info")]
    if len(dist_info) != 1:
        fail("Expected one DeepGEMM wheel metadata directory, found {}".format(dist_info))
    ctx.symlink("wheel/deep_gemm/csrc", "csrc")
    ctx.symlink("wheel/deep_gemm", "deep_gemm")
    ctx.symlink(dist_info[0].get_child("LICENSE"), "LICENSE")
    ctx.file("BUILD.bazel", ctx.attr.build_file_content)

_deep_gemm_wheel_sources = repository_rule(
    implementation = _deep_gemm_wheel_sources_impl,
    attrs = {
        "url": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
        "build_file_content": attr.string(mandatory = True),
    },
)

def blackwell_repositories():
    _wheel_lock(name = "rtp_blackwell_wheel_lock", url = DEEP_GEMM_WHEEL.url, sha256 = DEEP_GEMM_WHEEL.sha256)
    if CUDA_EXTENSION == None:
        return

    git_repository(
        name = "rtp_blackwell_kernels",
        remote = CUDA_EXTENSION.remote,
        commit = CUDA_EXTENSION.commit,
    )
    http_archive(
        name = "rtp_blackwell_elfutils",
        urls = [
            "https://codeload.github.com/{}/tar.gz/{}".format(ELFUTILS.repository, ELFUTILS.commit),
            "https://github.com/{}/archive/{}.tar.gz".format(ELFUTILS.repository, ELFUTILS.commit),
        ],
        type = "tar.gz",
        sha256 = ELFUTILS.sha256,
        strip_prefix = "elfutils-" + ELFUTILS.commit,
        build_file_content = _ELFUTILS_BUILD,
    )
    for name, source in BLACKWELL_SOURCES.items():
        repository_name = "rtp_blackwell_" + name
        build_file = _BUILD.format(srcs = repr(source.srcs))
        if hasattr(source, "wheel_url"):
            _deep_gemm_wheel_sources(
                name = repository_name,
                url = source.wheel_url,
                sha256 = source.sha256,
                build_file_content = build_file,
            )
            continue
        http_archive(
            name = repository_name,
            urls = [
                "https://codeload.github.com/{}/tar.gz/{}".format(source.repository, source.commit),
                "https://github.com/{}/archive/{}.tar.gz".format(source.repository, source.commit),
            ],
            type = "tar.gz",
            sha256 = source.sha256,
            strip_prefix = source.repository.split("/")[-1] + "-" + source.commit,
            build_file_content = build_file,
        )
