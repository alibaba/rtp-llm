"""Pinned, relocatable GCC for CUDA13 x86 runtime JIT compilation."""

_PACKAGES = [
    ("gcc-toolset-12-binutils-2.38-17.al8.x86_64.rpm", "fb295043beda188ee231bdca302f70e679ab25e5f6adc171403ff9c3dfa487b0"),
    ("gcc-toolset-12-binutils-gold-2.38-17.al8.x86_64.rpm", "0e6d15bd607c075685149518e4930c63c7e75f48d38fafb52ce4f378e389ce50"),
    ("gcc-toolset-12-gcc-12.3.0-1.2.al8.x86_64.rpm", "22ac28cd52861dfeb3a4962c8a201c56d94f78b11b4d668f0724047877665d13"),
    ("gcc-toolset-12-gcc-c++-12.3.0-1.2.al8.x86_64.rpm", "267ec7504458a4d7698da8c48f17f83c3c83cffb95a741956648613a73f756e0"),
    ("gcc-toolset-12-libstdc++-devel-12.3.0-1.2.al8.x86_64.rpm", "f1fb1815a956b49fca9cc728ce04e45f6e2abc966cfef6f02454ba35289da917"),
]

def _cuda13_jit_gcc_impl(ctx):
    rpm2cpio = ctx.which("rpm2cpio")
    cpio = ctx.which("cpio")
    if not rpm2cpio or not cpio:
        fail("Fetching cuda13_jit_gcc requires rpm2cpio and cpio on the Bazel host")
    for filename, sha256 in _PACKAGES:
        ctx.download(
            url = [
                "https://mirrors.aliyun.com/alinux/3/updates/x86_64/Packages/" + filename,
            ],
            output = filename,
            sha256 = sha256,
        )
        result = ctx.execute([
            "bash", "-c",
            'set -euo pipefail; "$1" "$2" | "$3" -idm --quiet --no-absolute-filenames',
            "extract-rpm", str(rpm2cpio), str(ctx.path(filename)), str(cpio),
        ])
        if result.return_code:
            fail("Cannot unpack %s: %s" % (filename, result.stderr))
        ctx.delete(filename)

    result = ctx.execute([
        "bash", "-c",
        "set -euo pipefail; chmod -R u+w opt/rh/gcc-toolset-12; mv opt/rh/gcc-toolset-12/root/usr toolchain",
    ])
    if result.return_code:
        fail("Cannot relocate GCC: " + result.stderr)
    ctx.delete("opt")
    ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])
py_library(
    name = "runtime",
    data = glob([
        "toolchain/bin/**",
        "toolchain/include/**",
        "toolchain/lib/**",
        "toolchain/lib64/**",
        "toolchain/libexec/**",
    ]),
)
""")

cuda13_jit_gcc_repository = repository_rule(
    implementation = _cuda13_jit_gcc_impl,
)
