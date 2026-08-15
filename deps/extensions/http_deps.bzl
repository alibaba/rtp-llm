load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load(":mirror.bzl", "rtp_github_archive_urls", "rtp_mirror_urls")

# Repo names created by this file; `scripts/rtpcli bazel mod-tidy` keeps root use_repo in sync.
HTTP_DEPS_EXPORTS = [
    "arm_compute",
    "boost",
    "bazel_skylib",
    "hedron_compile_commands",
    "jsoncpp_git",
    "hf3fs_rpm",
    "remote_kv_cache_manager_client_rpm",
    "remote_kv_cache_manager_server",
    "rules_pkg",
]

def http_deps():
    http_archive(
        name = "rules_pkg",
        urls = rtp_mirror_urls("archives/github.com/bazelbuild/rules_pkg/rules_pkg-0.6.0.tar.gz"),
        sha256 = "62eeb544ff1ef41d786e329e1536c1d541bb9bcad27ae984d57f18f314018e66",
    )

    http_archive(
        name = "bazel_skylib",
        sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
        urls = rtp_mirror_urls("archives/github.com/bazelbuild/bazel-skylib/bazel-skylib-1.5.0.tar.gz"),
    )

    # boost / jsoncpp: transitive deps of vipserver.
    http_archive(
        name = "boost",
        sha256 = "882b48708d211a5f48e60b0124cf5863c1534cd544ecd0664bb534a4b5d506e9",
        urls = rtp_mirror_urls("archives/boost_1_70_0.tar.gz"),
        strip_prefix = "boost_1_70_0",
        build_file = "@rtp_llm//3rdparty/boost:boost.BUILD",
        patches = ["@rtp_llm//patches/boost:boost.patch"],
    )

    http_archive(
        name = "jsoncpp_git",
        sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
        urls = rtp_mirror_urls("archives/jsoncpp-1.8.4.tar.gz"),
        strip_prefix = "jsoncpp-1.8.4",
        build_file = "@rtp_llm//3rdparty/jsoncpp:jsoncpp.BUILD",
    )

    http_archive(
        name = "arm_compute",
        sha256 = "6d7aebfa9be74d29ecd2dbeb17f69e00c667c36292401f210121bf26a30b38a5",
        urls = rtp_github_archive_urls("ARM-software", "ComputeLibrary", "v24.04"),
        strip_prefix = "ComputeLibrary-24.04",
    )

    http_archive(
        # Hedron's Compile Commands Extractor for Bazel
        name = "hedron_compile_commands",
        urls = rtp_github_archive_urls("hedronvision", "bazel-compile-commands-extractor", "4f28899228fb3ad0126897876f147ca15026151e"),
        strip_prefix = "bazel-compile-commands-extractor-4f28899228fb3ad0126897876f147ca15026151e",
        sha256 = "658122cfb1f25be76ea212b00f5eb047d8e2adc8bcf923b918461f2b1e37cdf2",
    )

    http_file(
        name = "hf3fs_rpm",
        urls = rtp_mirror_urls("package/3fs/hf3fs-1.3.0-1.alios7.x86_64.rpm"),
        sha256 = "dd375f794557a1135934b40b23a7435569644922c5c7116cb69dd36f699ad5a4",
    )

    http_file(
        name = "remote_kv_cache_manager_client_rpm",
        urls = rtp_mirror_urls("package/kvcm/kv-cache-manager-client-2026_04_29_14_29.rpm"),
        sha256 = "8a50e27c6c009bb2e9d55c7ff44ccef53268cc0b67559b95fd7e22221f1e9600",
    )

    http_archive(
        name = "remote_kv_cache_manager_server",
        urls = rtp_mirror_urls("package/kvcm/kv_cache_manager_server_2026_04_29_14_32.tar.gz"),
        sha256 = "6808080358f137c78205495b70b560261d59abff6eeddafc861e7511104c5b1a",
        build_file_content = """
exports_files(["bin/kv_cache_manager_bin"])
        """,
    )
