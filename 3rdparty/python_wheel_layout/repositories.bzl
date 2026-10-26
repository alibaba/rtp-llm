"""Preserve pip's shared directory layout for split CUDA 13 distributions."""

def _cuda13_wheel_layout_impl(ctx):
    # Bazel normally extracts each wheel into a different repository. Cutlass
    # 4.7 looks up its MLIR extensions and CUDA runtime relative to __file__;
    # FlashInfer's per-SM cache wheels likewise share one Python package.
    # Recreate the ordinary pip layout without changing vendor code or kernels.
    for wheel in ctx.attr.wheels:
        archive = ctx.path(wheel).basename + ".zip"
        ctx.symlink(ctx.path(wheel), archive)
        ctx.extract(archive, output = "site-packages")
        # Wheel .data/{purelib,platlib} entries must be relocated just as pip does.
        parts = ctx.path(wheel).basename.split("-")
        for scheme in ["purelib", "platlib"]:
            source = ctx.path("site-packages/{}-{}.data/{}".format(parts[0], parts[1], scheme))
            if source.exists:
                result = ctx.execute(["cp", "-a", str(source) + "/.", str(ctx.path("site-packages"))])
                if result.return_code:
                    fail("wheel layout relocation failed: " + result.stderr)
    ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])
py_library(
    name = "layout",
    srcs = glob(["site-packages/**/*.py"], exclude = ["**/*.data/**"]),
    data = glob(["site-packages/**/*"], exclude = ["**/*.py", "**/*.pyc", "**/*.data/**"]),
    imports = ["site-packages", "site-packages/nvidia_cutlass_dsl/dsl_packages"],
)
""")

_cuda13_wheel_layout = repository_rule(
    implementation = _cuda13_wheel_layout_impl,
    attrs = {"wheels": attr.label_list(allow_files = [".whl"])},
)

def cuda13_wheel_layout():
    """Use the exact wheels already verified by the CUDA13 requirements lock."""
    wheels = [
        "@pip_gpu_cuda13_torch_nvidia_cutlass_dsl_libs_core//:nvidia_cutlass_dsl_libs_core-4.7.1-py3-none-any.whl",
        "@pip_gpu_cuda13_torch_nvidia_cutlass_dsl_libs_base//:nvidia_cutlass_dsl_libs_base-4.7.1-cp310-cp310-manylinux_2_28_x86_64.whl",
        "@pip_gpu_cuda13_torch_nvidia_cutlass_dsl_libs_cu13//:nvidia_cutlass_dsl_libs_cu13-4.7.1-cp310-cp310-manylinux_2_28_x86_64.whl",
    ]
    for suffix in ["", "_sm80", "_sm90a", "_sm100a", "_sm103a", "_sm120f"]:
        wheels.append(
            "@pip_gpu_cuda13_torch_flashinfer_jit_cache{0}//:flashinfer_jit_cache{0}-0.7.2+36684fe4.cu132-cp39-abi3-manylinux_2_28_x86_64.whl".format(suffix),
        )
    _cuda13_wheel_layout(name = "cuda13_wheel_layout", wheels = wheels)
