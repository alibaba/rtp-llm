"""Pinned, namespaced FlashInfer MoE sidecar; does not replace flashinfer."""

def _flashinfer_opensource_impl(ctx):
    ctx.download_and_extract(
        url = "https://files.pythonhosted.org/packages/32/3f/3c8dbde51499902346f8f38fcdc3f92db3de25b1cdd11361f2f4dd90fd83/flashinfer_python-0.7.1-py3-none-any.whl",
        sha256 = "69302ed6e91314a85a4392324becc194594a782ba181646d25d5c305e1b7e8c4",
        type = "zip",
        output = "upstream",
    )
    result = ctx.execute([
        ctx.attr.python_interpreter,
        str(ctx.path(ctx.attr.relocator)),
        "--source-site", str(ctx.path("upstream")),
        "--output-site", str(ctx.path("site-packages")),
        "--source-wheel-sha256", "69302ed6e91314a85a4392324becc194594a782ba181646d25d5c305e1b7e8c4",
    ], timeout = 600)
    if result.return_code:
        fail("FlashInfer sidecar relocation failed:\n" + result.stdout + result.stderr)
    ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])
py_library(
    name = "pkg",
    srcs = glob(["site-packages/**/*.py"]),
    data = glob(["site-packages/**/*"], exclude = ["site-packages/**/*.py"]),
    imports = ["site-packages"],
)
""")

flashinfer_opensource_repository = repository_rule(
    implementation = _flashinfer_opensource_impl,
    attrs = {
        "relocator": attr.label(
            default = Label("@flashinfer_opensource_rules//:relocate.py"),
            allow_single_file = True,
        ),
        "python_interpreter": attr.string(default = "/opt/conda310/bin/python3"),
    },
)
