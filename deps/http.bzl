"""@rtp_deps OSS version — http_archive / http_file deps with public URLs.

Used by a fresh github.com/alibaba/rtp-llm clone (no internal_source). Internal
monorepo overrides @rtp_deps to ../internal_source/deps via
`--override_repository=rtp_deps=../internal_source/deps` in
internal_source/.internal_bazelrc, which loads the internal-mirror version
of these entries plus internal-only accelerator and transport dependencies.

Adding a new shared entry: also add it to internal_source/deps/http.bzl with
the internal-mirror URL. The two files drift independently; sha256 should
match across both URLs when the artifact is the same mirror of the same
upstream.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

def clean_dep(dep):
    return str(Label(dep))

def http_deps():
    # Bazel build tooling
    http_archive(
        name = "bazel_skylib",
        sha256 = "cd55a062e763b9349921f0f5db8c3933288dc8ba4f76dd9416aac68acee3cb94",
        urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz"],
    )

    http_archive(
        name = "hedron_compile_commands",
        sha256 = "658122cfb1f25be76ea212b00f5eb047d8e2adc8bcf923b918461f2b1e37cdf2",
        strip_prefix = "bazel-compile-commands-extractor-4f28899228fb3ad0126897876f147ca15026151e",
        urls = ["https://github.com/hedronvision/bazel-compile-commands-extractor/archive/4f28899228fb3ad0126897876f147ca15026151e.tar.gz"],
    )

    # hf3fs_rpm — consumed by //3rdparty/3fs:hf3fs_files (rpm_library).
    # Required by //:th_transformer on CUDA + ROCm analysis graphs.
    http_file(
        name = "hf3fs_rpm",
        urls = ["https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/3fs/hf3fs-1.3.0-1.alios7.x86_64.rpm"],
        sha256 = "dd375f794557a1135934b40b23a7435569644922c5c7116cb69dd36f699ad5a4",
    )

    # remote_kv_cache_manager_client_rpm — consumed by
    # //3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client
    # (rpm_library wraps it into a cc_library). Unconditionally pulled by
    # //rtp_llm/cpp/cache/connector/remote_connector:client on the
    # //:th_transformer dep graph for all platforms.
    http_file(
        name = "remote_kv_cache_manager_client_rpm",
        urls = [
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/kvcm/kv-cache-manager-client-2026_04_29_14_29.rpm",
        ],
        sha256 = "8a50e27c6c009bb2e9d55c7ff44ccef53268cc0b67559b95fd7e22221f1e9600",
    )

    # remote_kv_cache_manager_server — tarball with bin/kv_cache_manager_bin.
    # Consumed by smoke tests that exercise the remote kv cache manager path.
    http_archive(
        name = "remote_kv_cache_manager_server",
        urls = [
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/kvcm/kv_cache_manager_server_2026_04_29_14_32.tar.gz",
        ],
        sha256 = "6808080358f137c78205495b70b560261d59abff6eeddafc861e7511104c5b1a",
        build_file_content = "exports_files([\"bin/kv_cache_manager_bin\"])\n",
    )
