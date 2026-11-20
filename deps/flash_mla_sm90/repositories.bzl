"""Isolated Hopper FlashMLA; the default release wheel remains for SM10x."""

def _flash_mla_sm90_impl(ctx):
    ctx.download_and_extract(
        url = "https://search-ad.oss-cn-hangzhou-zmf-internal.aliyuncs.com/rtp-llm-deps/dsv41/sm90/9241ae3/flash_mla-1.0.0%2B9241ae3-cp310-cp310-linux_x86_64.whl",
        sha256 = "67bba01c854fe3b06397c297d49724e16029984a37124d05047a78ff48e7c44f",
        type = "zip",
        output = "upstream",
    )
    # The extension retains its basename (PyInit_cuda), but lives in a distinct
    # Python package. Only absolute Python imports require relocation.
    result = ctx.execute([ctx.attr.python_interpreter, "-c", """
from pathlib import Path
import hashlib
import shutil
src = Path('upstream/flash_mla')

# The pinned Hopper wheel computes num_sm_parts as
# num_sms / s_q / (h_q / 64).  Its SM90 kernel already maps s_q to grid.y, so
# that first division changes the split-K reduction order for speculative
# multi-query decode. Keep it removed so single/multi-query plans agree, and
# use a 32-SM planner budget to reduce partial-output traffic for small-batch
# decode. This changes the split plan, not the number of SMs the GPU can use.
# The reduction tree can differ from the original hardware-SM-count plan;
# numerical equivalence is tolerance-based, not bitwise against that plan.
# Keep the measured MLA32 package reproducible without shipping a private
# binary wheel. The pinned wheel, unique instruction sequence and output
# digest make this fail closed if the input or patch changes.
cuda_so = next(src.glob('cuda*.so'))
binary = cuda_so.read_bytes()
needle = bytes.fromhex('8b808001000099f7fd85db0f49cb')
replacement = bytes.fromhex('b8200000009090909085db0f49cb')
if binary.count(needle) != 1:
    raise RuntimeError('unexpected FlashMLA SM90 get_meta binary layout')
patched = binary.replace(needle, replacement, 1)
if hashlib.sha256(patched).hexdigest() != 'c33eb8fda638562b469e55c66c930284e316f40b58c97407c69abab133e1ca2e':
    raise RuntimeError('unexpected FlashMLA SM90 MLA32 output digest')
cuda_so.write_bytes(patched)

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
