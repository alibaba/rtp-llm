load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def xgrammar_deps():
    # v0.2.8: Kimi K3 XML structural tags and bounded AnyText/AnyTokens.
    new_git_repository(
        name = "xgrammar",
        remote = "https://github.com/mlc-ai/xgrammar.git",
        commit = "97787376faee5ed8466cfad57c99855e4ce2f6aa",
        init_submodules = False,
        build_file = str(Label("@rtp_llm//3rdparty/xgrammar:xgrammar.BUILD")),
    )
