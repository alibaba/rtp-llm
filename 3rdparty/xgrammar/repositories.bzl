load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def xgrammar_deps():
    # xgrammar 0.2.7 runtime, aligned with the Python wheel.
    new_git_repository(
        name = "xgrammar",
        remote = "https://github.com/mlc-ai/xgrammar.git",
        commit = "0b586e9bfa3ba60924bf2248b9c0019e83eb2ab6",
        init_submodules = False,
        patch_cmds = [
            "git submodule update --init --depth=1 3rdparty/picojson",
        ],
        build_file = str(Label("@rtp_llm//3rdparty/xgrammar:xgrammar.BUILD")),
    )
