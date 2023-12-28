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
        sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
        urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.5.0/bazel-skylib-1.5.0.tar.gz"],
    )

    http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
        strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
        urls = [
           "https://github.com/bazelbuild/rules_closure/archive/refs/tags/0.12.0.zip",
        ]
    )

    http_archive(
        name = "torch_2.1_py310_cpu",
        sha256 = "5077921fc2b54e69a534f3a9c0b98493c79a5547c49d46f5e77e42da3610e011",
        urls = [
            "http://search-ad.oss-cn-hangzhou-zmf-internal.aliyuncs.com/rtp%2Ftorch210%2Ftorch-2.1.0%2Bcpu-cp310-cp310-linux_x86_64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )