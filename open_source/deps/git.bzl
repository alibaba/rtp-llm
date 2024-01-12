load('@bazel_tools//tools/build_defs/repo:git.bzl', "git_repository", "new_git_repository")

def git_deps():
    git_repository(
        name = "rules_python",
        remote = "https://github.com/bazelbuild/rules_python.git",
        patches = ["//patches/rules_python:0001-fix-triton-and-pypi-wheel.patch"],
        commit = "5eb0de810f76f16ab8a909953c1b235051536686",
        shallow_since = "1611475624 +1100",
    )

    new_git_repository(
        name = "cutlass",
        remote = "https://github.com/NVIDIA/cutlass.git",
        commit = "8783c41851cd3582490e04e69e0cd756a8c1db7f",
        build_file = str(Label("//3rdparty/cutlass:cutlass.BUILD")),
    )

    git_repository(
        name = "com_google_googletest",
        remote = "https://github.com/google/googletest.git",
        commit = "1a9f2cf450187ff4e52ad8fc6dae4aaac6924c7b",
        shallow_since = "1640057570 +0800",
    )