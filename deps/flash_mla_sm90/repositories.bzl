"""Isolated Hopper FlashMLA; the default release wheel remains for SM10x."""

def _flash_mla_sm90_impl(ctx):
    ctx.download_and_extract(
        url = "https://rtp-maga.oss-cn-zhangjiakou.aliyuncs.com/miji/0430/flash_mla-1.0.0%2B9241ae3-cp310-cp310-linux_x86_64.whl",
        sha256 = "67bba01c854fe3b06397c297d49724e16029984a37124d05047a78ff48e7c44f",
        type = "zip",
        output = "upstream",
    )
    # The extension retains its basename (PyInit_cuda), but lives in a distinct
    # Python package. Only absolute Python imports require relocation.
    result = ctx.execute([ctx.attr.python_interpreter, "-c", """
from pathlib import Path
import shutil
src = Path('upstream/flash_mla')

# The pinned Hopper wheel computes num_sm_parts as
# num_sms / s_q / (h_q / 64).  Its SM90 kernel already maps s_q to grid.y, so
# that first division only changes the split-K reduction order for speculative
# multi-query decode.  Patch the three x86 instructions `cltd; idiv %ebp` to
# NOPs, preserving the exact single-query split plan without rebuilding or
# carrying a private binary wheel.  The surrounding bytes and wheel sha256
# make this fail closed if upstream changes.
cuda_so = next(src.glob('cuda*.so'))
binary = cuda_so.read_bytes()
needle = bytes.fromhex('8b808001000099f7fd85db0f49cb')
replacement = bytes.fromhex('8b808001000090909085db0f49cb')
if binary.count(needle) != 1:
    raise RuntimeError('unexpected FlashMLA SM90 get_meta binary layout')
cuda_so.write_bytes(binary.replace(needle, replacement, 1))

dst = Path('site-packages/flash_mla_sm90')
shutil.copytree(src, dst)
for path in dst.rglob('*.py'):
    path.write_text(path.read_text().replace('from flash_mla.', 'from flash_mla_sm90.').replace('import flash_mla.', 'import flash_mla_sm90.'))
"""])
    if result.return_code:
        fail("FlashMLA SM90 relocation failed: " + result.stderr)
    ctx.file("BUILD.bazel", """
load("@rules_python//python:packaging.bzl", "py_package")
package(default_visibility = ["//visibility:public"])
py_library(
    name = "pkg",
    srcs = glob(["site-packages/**/*.py"]),
    data = glob(["site-packages/**/*.so"]),
    imports = ["site-packages"],
)
py_package(
    name = "wheel_package",
    deps = [":pkg"],
    packages = ["site-packages/flash_mla_sm90"],
)
""")

flash_mla_sm90_repository = repository_rule(
    implementation = _flash_mla_sm90_impl,
    attrs = {"python_interpreter": attr.string(default = "/opt/conda310/bin/python3")},
)
