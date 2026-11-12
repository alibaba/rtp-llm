load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def xgrammar_deps():
    # XGrammar v0.2.9: keep native grammar and Python wheel on one release.
    new_git_repository(
        name = "xgrammar",
        remote = "https://github.com/mlc-ai/xgrammar.git",
        commit = "8f9a1279fdbed47739bd75b4099d1d059a1ce3e0",
        init_submodules = False,
        build_file = str(Label("@rtp_llm//3rdparty/xgrammar:xgrammar.BUILD")),
    )
