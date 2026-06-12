load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def xgrammar_deps():
    # xgrammar with structural-tag max_tokens support for reasoning budgets.
    new_git_repository(
        name = "xgrammar",
        remote = "https://github.com/izhuhaoran/xgrammar.git",
        commit = "ac8a29c0212c2e19484b558f1fc029fa7f973513",
        init_submodules = False,
        build_file = str(Label("@rtp_llm//3rdparty/xgrammar:xgrammar.BUILD")),
    )
