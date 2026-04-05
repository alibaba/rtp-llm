load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

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
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/kvcm/kv-cache-manager-client-2026_04_02_12_08.rpm",
        ],
        sha256 = "52e8f29e1de1099fa90665443f774a7cbd7b3fa86827a3693273fdd6fc57773e",
    )

    http_archive(
        name = "remote_kv_cache_manager_server",
        urls = [
            "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/package/kvcm/kv_cache_manager_server_2026_02_28_11_36.tar.gz",
        ],
        sha256 = "757eaec92b45a156ae02bae2000db54d767538c572276269ebc803c1513bb3f2",
        build_file_content = """
exports_files(["bin/kv_cache_manager_bin"])
        """,
    )
