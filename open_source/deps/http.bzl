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
        ]
    )

    http_archive(
        name = "torch_2.1_py310_cpu",
        sha256 = "5077921fc2b54e69a534f3a9c0b98493c79a5547c49d46f5e77e42da3610e011",
        urls = [
            "https://download.pytorch.org/whl/cpu/torch-2.1.0%2Bcpu-cp310-cp310-linux_x86_64.whl#sha256=5077921fc2b54e69a534f3a9c0b98493c79a5547c49d46f5e77e42da3610e011",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "xfastertransformer_devel_icx",
        sha256 = "dfd1714815d38dfea89532365fbe36d502f0bb3baf37c0b472c16743a8cbe352",
        urls = [
            "https://files.pythonhosted.org/packages/b9/0a/9b2da7873a1bada71c3d686ec7ae3a01a1b3864f08b9d07aca7e9e841615/xfastertransformer_devel_icx-1.8.1.1-py3-none-any.whl",
            # "https://mirrors.aliyun.com/pypi/packages/99/96/ec754dbc62cc0216ac5e95e91f3d9ad43035fbade1d2c35326b7bd267a1d/xfastertransformer_devel_icx-1.7.2-py3-none-any.whl",
        ],
        type = "zip",
        build_file = clean_dep("//3rdparty/xft:BUILD"),
    )

    http_archive(
        name = "xfastertransformer_devel",
        sha256 = "2344c92cbec175602895bfc76db862a7f724ab9ae0e4aa89bc1b462dfa25b2e9",
        urls = [
            "https://files.pythonhosted.org/packages/a9/67/4133273051133b5848fa29a7da78528c85a013372ee8ca9b90cbc51c4ae0/xfastertransformer_devel-1.8.1.1-py3-none-any.whl",
            # "https://mirrors.aliyun.com/pypi/packages/70/f9/3d4b31a489c733bbbe37cdc76a4ccf7b71ee0895c038a90d8f722d08247e/xfastertransformer_devel-1.7.2-py3-none-any.whl",
        ],
        type = "zip",
        build_file = clean_dep("//3rdparty/xft:BUILD"),
    )

    http_archive(
        name = "torch_2.3_py310_cpu_aarch64",
        sha256 = "ad89e5a7dfaa96a3c054aaf644003b3e72cd34260b1288fdf4fb638eb9b15795",
        urls = [
            # This is a custom build of torch 2.3.0 for aarch64 with ACL 24.04 to workaround FP16 issue with ACL 23.08
            "https://github.com/TianyuLi0/rtp-llm/raw/pytorch_2.3.0_aarch64_acl_24.04/3rdparty/acl/torch-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        ],
        type = "zip",
        build_file = clean_dep("//:BUILD.pytorch"),
    )

    http_archive(
        name = "arm_compute",
        sha256 = "6d7aebfa9be74d29ecd2dbeb17f69e00c667c36292401f210121bf26a30b38a5",
        urls = ["https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v24.04.tar.gz"],
        strip_prefix = "ComputeLibrary-24.04",
    )
