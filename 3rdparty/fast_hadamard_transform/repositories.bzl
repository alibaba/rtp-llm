load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def fast_hadamard_transform_deps():
    http_archive(
        name = "fast_hadamard_transform_src",
        urls = [
            "https://api.github.com/repos/Dao-AILab/fast-hadamard-transform/tarball/4ea722e434e3d4f2a14522341959ebdbe62be2de",
            "https://codeload.github.com/Dao-AILab/fast-hadamard-transform/legacy.tar.gz/4ea722e434e3d4f2a14522341959ebdbe62be2de",
            "https://gh-proxy.com/https://api.github.com/repos/Dao-AILab/fast-hadamard-transform/tarball/4ea722e434e3d4f2a14522341959ebdbe62be2de",
        ],
        type = "tar.gz",
        sha256 = "8d562f2696861521c557e6c920e0617b4627608430d0949b323c29c88f554933",
        strip_prefix = "Dao-AILab-fast-hadamard-transform-4ea722e",
        build_file = "//3rdparty/fast_hadamard_transform:fast_hadamard_transform.BUILD",
        patches = ["//3rdparty/fast_hadamard_transform:embedded_binding.patch"],
        patch_args = ["-p1"],
    )
