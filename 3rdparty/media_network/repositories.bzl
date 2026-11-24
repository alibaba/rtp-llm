"""Network libraries linked privately into the ViT download extension."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def media_network_deps():
    http_archive(
        name = "media_curl",
        build_file = "//3rdparty/media_network:curl.BUILD",
        sha256 = "8dadd7d2536a314f982ad05a651b713f426bf7d02cc2c76cffb9796f2218e6c5",
        strip_prefix = "curl-8.23.0",
        urls = ["https://curl.se/download/curl-8.23.0.tar.xz"],
    )
    http_archive(
        name = "media_cares",
        build_file = "//3rdparty/media_network:cares.BUILD",
        sha256 = "7d935790e9af081c25c495fd13c2cfcda4792983418e96358ef6e7320ee06346",
        strip_prefix = "c-ares-1.34.5",
        urls = ["https://github.com/c-ares/c-ares/releases/download/v1.34.5/c-ares-1.34.5.tar.gz"],
    )
    http_archive(
        name = "media_boringssl",
        build_file = "//3rdparty/media_network:boringssl.BUILD",
        sha256 = "d1c599485fd1919d75ea2925af5fff81c1d5b21ab2f0d41fee1f788b1d917159",
        strip_prefix = "boringssl-0.20260616.0",
        urls = ["https://github.com/google/boringssl/releases/download/0.20260616.0/boringssl-0.20260616.0.tar.gz"],
    )
    http_archive(
        name = "media_brotli",
        build_file = "//3rdparty/media_network:brotli.BUILD",
        sha256 = "816c96e8e8f193b40151dad7e8ff37b1221d019dbcb9c35cd3fadbfe6477dfec",
        strip_prefix = "brotli-1.2.0",
        urls = ["https://github.com/google/brotli/archive/refs/tags/v1.2.0.tar.gz"],
    )
    http_archive(
        name = "media_zstd",
        build_file = "//3rdparty/media_network:zstd.BUILD",
        sha256 = "eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3",
        strip_prefix = "zstd-1.5.7",
        urls = ["https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"],
    )
