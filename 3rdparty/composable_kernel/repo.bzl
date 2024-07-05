BUILD_CONTENT = """
load("@//:def.bzl", "rocm_copts")

alias(
    name = "ck_headers",
    actual = "@composable_kernel_archive//:ck_headers",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_fmha_example",
    srcs = glob([
        "fmha_example_gen/*.cpp",
    ]),
    copts = rocm_copts(),
    deps = [
        "@composable_kernel_archive//:ck_fmha_example_headers",
    ],
    visibility = ["//visibility:public"],
)
"""

def _ck_repo_impl(repository_ctx):
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "python")
    generate_py = Label("@composable_kernel_archive//:example/ck_tile/01_fmha/generate.py")
    user_args = repository_ctx.os.environ.get("CK_FMHA_GEN_ARGS", " ")
    args = [python_bin, generate_py, '-o', 'fmha_example_gen']
    args.extend([arg.strip() for arg in user_args.split(' ')])
    result = repository_ctx.execute(args)
    if result.return_code :
        fail(' '.join([str(arg) for arg in args]), "\n", result.stderr)
    repository_ctx.file("BUILD", BUILD_CONTENT)

ck_repo = repository_rule(
    implementation = _ck_repo_impl,
    environ = [
        "PYTHON_BIN_PATH",
        "CK_FMHA_GEN_ARGS",
    ],
)


