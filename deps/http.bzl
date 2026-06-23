load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# TODO(pip_unify Phase 5): shared http_archive entries (rules_pkg, bazel_skylib,
# io_bazel_rules_closure, torch_2.8_py310_cuda, torch_rocm, aiter,
# arm_compute, hedron_compile_commands) are also declared in
# internal_source/deps/http.bzl with matching sha256 but different URL sources.
# Consolidate here as multi-URL lists (artlab mirror first, public URL fallback);
# requires per-config `bazel build` verification before landing to catch URL
# availability regressions.

def clean_dep(dep):
    return str(Label(dep))

def http_deps():
    http_archive(
        name = "rules_pkg",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.6.0/rules_pkg-0.6.0.tar.gz",
        ],
        sha256 = "62eeb544ff1ef41d786e329e1536c1d541bb9bcad27ae984d57f18f314018e66",
    )

    http_archive(
        name = "bazel_skylib",
        sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
        urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz"],
    )

    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
        strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
        urls = [
            "https://github.com/bazelbuild/rules_closure/archive/refs/tags/0.12.0.zip",
        ],
    )

    http_archive(
        name = "torch_2.8_py310_cuda",
        sha256 = "54d240b5d3b1f9075d4ee6179675a22c1974f7bef1885d134c582678d5180cd3",
        urls = [
            "https://download.pytorch.org/whl/cu129/torch-2.8.0%2Bcu129-cp310-cp310-manylinux_2_28_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("@rtp_llm//:BUILD.pytorch"),
    )

    http_archive(
        name = "torch_rocm",
        sha256 = "521d1febc9bfebe44fb321727ad550dcaf05900dd917b20bed52fb307f43bf3a",
        urls = [
            "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/torch/torch-2.9.1%2Bgit7e1940d-cp310-cp310-linux_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("@rtp_llm//:BUILD.pytorch"),
    )

    http_archive(
        name = "torch_2.9_py310_cuda_arm",
        sha256 = "37780eb80e4319d6e004ea9597353da0b3947681866d7adff4757ece164a5cd9",
        urls = [
            "https://download.pytorch.org/whl/cu129/torch-2.9.0%2Bcu129-cp310-cp310-manylinux_2_28_aarch64.whl",
        ],
        type = "zip",
        build_file = clean_dep("@rtp_llm//:BUILD.pytorch"),
    )

    # TODO(pip_unify Phase 5): aiter http_archive (C++ headers/runtime) is
    # 0.1.14rc1 while requirements_rocm.txt / requirements_lock_rocm.txt pin
    # 0.1.13.dev14. Unify both to the same version once the matching wheel URL
    # and sha256 are available, or regenerate the ROCm lockfile with 0.1.14rc1.
    # Until then, keep the archive at the newer version but note the ABI gap.
    http_archive(
        name = "aiter",
        sha256 = "83c6bf067f94f8ca901a7d0526c51835c615feae9e9299f0371daef53e55bdd2",
        urls = [
            "https://sinian-metrics-platform.oss-cn-hangzhou.aliyuncs.com/kis/AMD/RTP/aiter-0.1.14rc1.dev41%2Bgc39217100.d20260519-cp310-cp310-linux_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("@rtp_llm//:BUILD.aiter"),
    )

    http_archive(
        name = "arm_compute",
        sha256 = "6d7aebfa9be74d29ecd2dbeb17f69e00c667c36292401f210121bf26a30b38a5",
        urls = ["https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v24.04.tar.gz"],
        strip_prefix = "ComputeLibrary-24.04",
    )

    http_archive(
        # Hedron's Compile Commands Extractor for Bazel
        name = "hedron_compile_commands",
        urls = ["https://github.com/hedronvision/bazel-compile-commands-extractor/archive/4f28899228fb3ad0126897876f147ca15026151e.tar.gz"],
        strip_prefix = "bazel-compile-commands-extractor-4f28899228fb3ad0126897876f147ca15026151e",
        sha256 = "658122cfb1f25be76ea212b00f5eb047d8e2adc8bcf923b918461f2b1e37cdf2",
    )

    http_file(
        name = "hf3fs_rpm",
        urls = ["https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/3fs/hf3fs-1.3.0-1.alios7.x86_64.rpm"],
        sha256 = "dd375f794557a1135934b40b23a7435569644922c5c7116cb69dd36f699ad5a4",
    )

    http_file(
        name = "remote_kv_cache_manager_client_rpm",
        urls = [
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/kvcm/kv-cache-manager-client-2026_04_29_14_29.rpm",
        ],
        sha256 = "8a50e27c6c009bb2e9d55c7ff44ccef53268cc0b67559b95fd7e22221f1e9600",
    )

    http_archive(
        name = "remote_kv_cache_manager_server",
        urls = [
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/kvcm/kv_cache_manager_server_2026_04_29_14_32.tar.gz",
        ],
        sha256 = "6808080358f137c78205495b70b560261d59abff6eeddafc861e7511104c5b1a",
        build_file_content = """
exports_files(["bin/kv_cache_manager_bin"])
        """,
    )
