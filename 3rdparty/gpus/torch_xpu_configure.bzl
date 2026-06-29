"""Repository rule for torch XPU autoconfiguration.

`torch_xpu_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: The python binary path. Used to detect site-packages.

This rule creates a repository pointing at the system-installed PyTorch XPU
site-packages directory, detected automatically from the Python interpreter
on PATH or via `PYTHON_BIN_PATH`.
"""

load("//3rdparty/gpus:xpu_python_utils.bzl", "resolve_venv_python")

def _torch_xpu_configure_impl(repository_ctx):
    # Check if XPU torch is actually available; if not, create a dummy repo
    # so CUDA/ROCm builds don't fail.
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "")
    if not python_bin:
        python_bin = repository_ctx.which("python3")
        if python_bin == None:
            # No python3 — create dummy and return
            repository_ctx.file("BUILD.bazel", "# dummy torch_xpu repo (no python3 found)\n")
            return
        python_bin = str(python_bin)

    # Resolve symlinked python to venv python so import torch works
    python_bin = resolve_venv_python(repository_ctx, python_bin)

    # Check if torch.xpu is available in this Python
    check = repository_ctx.execute([
        python_bin, "-c", "import torch; assert hasattr(torch, 'xpu')",
    ])
    if check.return_code != 0:
        # When XPU build is explicitly requested, fail instead of silently
        # creating a stub — avoids hard-to-diagnose link errors later.
        if repository_ctx.os.environ.get("TF_NEED_XPU", "0") == "1":
            fail("TF_NEED_XPU=1 but torch.xpu is not available in " + python_bin +
                 " (exit code " + str(check.return_code) + ")" +
                 "\nstdout: " + check.stdout +
                 "\nstderr: " + check.stderr)

        # torch.xpu not available — create minimal stub BUILD so
        # CUDA/ROCm builds don't fail on unresolvable dependencies.
        repository_ctx.file("BUILD.bazel", """
package(default_visibility = ["//visibility:public"])
cc_library(name = "torch")
cc_library(name = "torch_api")
cc_library(name = "torch_libs")
""")
        repository_ctx.file("torch/lib/.empty", "")
        return

    # Auto-detect site-packages from the actual torch installation path.
    # Using torch.__file__ instead of site.getsitepackages() avoids mismatches
    # in venv/system-site-packages or custom sys.path configurations.
    result = repository_ctx.execute([
        python_bin,
        "-c",
        "import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))",
    ])
    if result.return_code != 0:
        fail("Failed to detect site-packages from torch.__file__: " + result.stderr)

    site_packages = result.stdout.strip()

    # When XPU is explicitly requested, verify the actual torch XPU runtime
    # libraries are present (hasattr(torch, 'xpu') is true even on CPU-only
    # torch). Fail fast at config time with a clear message instead of letting
    # the build fail much later at link time.
    if repository_ctx.os.environ.get("TF_NEED_XPU", "0") == "1":
        torch_lib_dir = site_packages + "/torch/lib"
        missing = []
        for so in ["libtorch_xpu.so", "libc10_xpu.so"]:
            if not repository_ctx.path(torch_lib_dir + "/" + so).exists:
                missing.append(so)
        if missing:
            fail("TF_NEED_XPU=1 but required torch XPU libraries are missing " +
                 "from " + torch_lib_dir + ": " + ", ".join(missing) +
                 ". Install a torch build with XPU support (torch==*+xpu).")

    # List site-packages entries and symlink each one into the repo root,
    # reproducing the same layout as new_local_repository(path = site_packages).
    ls_result = repository_ctx.execute([
        python_bin, "-c",
        "import os, sys; print('\\n'.join(os.listdir(sys.argv[1])))",
        site_packages,
    ])
    if ls_result.return_code != 0:
        fail("Failed to list site-packages: " + ls_result.stderr)
    # Only symlink directories actually needed by BUILD.pytorch to reduce
    # repository rule I/O and invalidation surface.
    _needed = {"torch": True, "torch.libs": True, "torch.dist-info": True}
    for entry in ls_result.stdout.strip().split("\n"):
        if entry and (entry in _needed or entry.startswith("torch-")
                      or entry.startswith("torch_")):
            repository_ctx.symlink(site_packages + "/" + entry, entry)

    # Generate BUILD file from the provided build_file (overwrites any
    # symlinked BUILD that may exist in site-packages).
    build_file = repository_ctx.attr.build_file
    repository_ctx.symlink(build_file, "BUILD.bazel")

torch_xpu_configure = repository_rule(
    implementation = _torch_xpu_configure_impl,
    attrs = {
        "build_file": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "BUILD file for the torch XPU repository.",
        ),
    },
    environ = [
        "PYTHON_BIN_PATH",
        "TF_NEED_XPU",
    ],
    doc = "Auto-detects PyTorch XPU site-packages and creates a repository.",
)
