BUILD_CONTENT = """
load("@//:def.bzl", "rocm_copts")

alias(
    name = "ck_headers",
    actual = "@composable_kernel_archive//:ck_headers",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_fmha_example",
    srcs = ["@composable_kernel_archive//:ck_fmha_rmsnorm2d_libraries"],
    copts = rocm_copts(),
    deps = [
        "@composable_kernel_archive//:ck_fmha_example_headers",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_layernorm2d_example",
    srcs = glob([
        "layernorm2d_example_gen/*.cpp",
    ]),
    hdrs = glob([
        "layernorm2d_example_gen/*.hpp",
    ]),
    copts = rocm_copts(),
    deps = [
        "@composable_kernel_archive//:ck_layernorm2d_example_headers",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_rmsnorm2d_example",
    srcs = ["@composable_kernel_archive//:ck_fmha_rmsnorm2d_libraries"],
    copts = rocm_copts(),
    deps = [
        "@composable_kernel_archive//:ck_rmsnorm2d_example_headers",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ck_fused_moe_example",
    hdrs = glob(["example/ck_tile/15_fused_moe/**/*.hpp",]),
    copts = rocm_copts(),
    deps = [
        "@composable_kernel_archive//:ck_fused_moe_example_headers",
    ],
    visibility = ["//visibility:public"],
)
"""

def _ck_repo_impl(repository_ctx):
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "python")
    fmha_generate_py = Label("@composable_kernel_archive//:example/ck_tile/01_fmha/generate.py")
    fmha_user_args = repository_ctx.os.environ.get("CK_FMHA_GEN_ARGS", " ")
    fmha_args = [python_bin, fmha_generate_py, '-o', 'fmha_example_gen']
    fmha_args.extend([arg.strip() for arg in fmha_user_args.split(' ')])
    result = repository_ctx.execute(fmha_args)
    if result.return_code :
        fail(' '.join([str(arg) for arg in fmha_args]), "\n", result.stderr)
    repository_ctx.file("BUILD", BUILD_CONTENT)
    lynm_generate_py = Label("@composable_kernel_archive//:example/ck_tile/02_layernorm2d/generate.py")
    lynm_user_args = repository_ctx.os.environ.get("CK_LYNM_GEN_ARGS", " ")
    lynm_args = [python_bin, lynm_generate_py, '-a', 'fwd', '--gen_blobs', '-w', 'layernorm2d_example_gen']
    lynm_args.extend([arg.strip() for arg in lynm_user_args.split(' ')])
    result = repository_ctx.execute(lynm_args)
    if result.return_code :
        fail(' '.join([str(arg) for arg in lynm_args]), "\n", result.stderr)
    repository_ctx.file("BUILD", BUILD_CONTENT)

ck_repo = repository_rule(
    implementation = _ck_repo_impl,
    environ = [
        "PYTHON_BIN_PATH",
        "CK_FMHA_GEN_ARGS",
        "CK_LYNM_GEN_ARGS",
    ],
)