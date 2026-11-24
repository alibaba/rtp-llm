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
