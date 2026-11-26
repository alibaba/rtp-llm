"""Build the native operators and their JIT inputs in the installed wheel layout."""

load("@internal_impls//:impls.bzl", "CUDA_EXTENSION")
load(":repositories.bzl", "BLACKWELL_SOURCES", "ELFUTILS")

_BINDINGS = "//rtp_llm/models_py/bindings/cuda/blackwell:"
_KERNELS = "@rtp_blackwell_kernels//csrc/kernels/blackwell:"
_COMPONENTS = {
    "mhc": ["deep_gemm", "deep_jit", "cutlass_mhc", "fmt"],
    "sparse": ["flash_mla", "cutlass_mla"],
}

def _extensions_impl(ctx):
    outputs = {
        "mhc": ctx.actions.declare_directory("mega_mhc_isolated"),
        "sparse": ctx.actions.declare_file("rtp_fp8_sparse_decode.abi3.so"),
    }
    bindings = {
        "mhc": ctx.file._mhc_binding,
        "sparse": ctx.file._sparse_binding,
    }
    kernels = {
        "mhc": [ctx.file.mhc_runtime],
        "sparse": ctx.files.sparse_sources + ctx.files.sparse_headers,
    }
    sources = {name: target.files for target, name in ctx.attr.sources.items()}
    licenses = {name: target.files.to_list()[0] for target, name in ctx.attr.license_files.items()}
    builder = ctx.attr._builder[DefaultInfo].files_to_run
    for part, repositories in _COMPONENTS.items():
        args = ctx.actions.args()
        args.add(part)
        args.add("--output", outputs[part].path)
        args.add("--binding", bindings[part])
        inputs = [bindings[part]] + kernels[part]
        args.add("--kernel-dir", kernels[part][0].dirname)
        for name in repositories:
            args.add("--source", name + "=" + licenses[name].dirname)
        if part == "mhc":
            args.add("--format-header", ctx.file._format)
            for header in ctx.files.elfutils_headers:
                args.add("--elfutils-header", header)
            inputs.extend(ctx.files.elfutils_headers)
            inputs.append(ctx.file._format)
            for name, license in licenses.items():
                args.add("--license", name + "=" + license.path)
            inputs.extend(licenses.values())
        ctx.actions.run(
            executable = builder,
            arguments = [args],
            inputs = depset(inputs, transitive = [sources[name] for name in repositories]),
            outputs = [outputs[part]],
            mnemonic = "BlackwellExtension",
            progress_message = "Compiling {} for %{{label}}".format(part),
            use_default_shell_env = True,
        )

    mqa_adapter = ctx.actions.declare_file("fp8_mqa_adapter.py")
    ctx.actions.symlink(output = mqa_adapter, target_file = ctx.file.mqa_adapter)
    assets = outputs.values() + [ctx.file._check, mqa_adapter]
    manifest = ctx.actions.declare_file("glm53_extensions.json")
    args = ctx.actions.args()
    args.add("manifest")
    args.add("--output", manifest)
    for asset in assets:
        args.add("--asset", asset.path)
    pins = {}
    for name, src in BLACKWELL_SOURCES.items():
        pins[name] = {"repository": src.repository, "commit": src.commit, "sha256": src.sha256}
    pins["cuda_extension"] = {"repository": "foundation_models/cuda_extension", "commit": CUDA_EXTENSION.commit}
    pins["elfutils"] = {"repository": ELFUTILS.repository, "commit": ELFUTILS.commit, "sha256": ELFUTILS.sha256}
    args.add("--pins", json.encode(pins))
    ctx.actions.run(
        executable = builder,
        arguments = [args],
        inputs = assets,
        outputs = [manifest],
        mnemonic = "BlackwellManifest",
        use_default_shell_env = True,
    )
    files = assets + [manifest]
    return [DefaultInfo(files = depset(files), runfiles = ctx.runfiles(files = files))]

_extensions = rule(
    implementation = _extensions_impl,
    attrs = {
        "sources": attr.label_keyed_string_dict(),
        "license_files": attr.label_keyed_string_dict(),
        "mhc_runtime": attr.label(allow_single_file = [".h"]),
        "mqa_adapter": attr.label(allow_single_file = [".py"]),
        "sparse_sources": attr.label_list(allow_files = [".cpp", ".cu"]),
        "sparse_headers": attr.label_list(allow_files = [".h"]),
        "elfutils_headers": attr.label_list(allow_files = [".h"]),
        # PyTorch's headers, libraries and CUDAExtension must match the target wheel.
        "_builder": attr.label(default = _BINDINGS + "build_extension", executable = True, cfg = "target"),
        "_mhc_binding": attr.label(default = _BINDINGS + "mhc_binding.cpp", allow_single_file = True),
        "_sparse_binding": attr.label(default = _BINDINGS + "sparse_mla_registration.cpp", allow_single_file = True),
        "_format": attr.label(default = _BINDINGS + "format", allow_single_file = True),
        "_check": attr.label(default = "//rtp_llm/libs:glm53_extension_check.py", allow_single_file = True),
    },
)

def blackwell_extensions(name):
    if CUDA_EXTENSION == None:
        native.filegroup(name = name)
        return
    _extensions(
        name = name,
        sources = {"@rtp_blackwell_%s//:sources" % repo: repo for repo in BLACKWELL_SOURCES},
        license_files = {"@rtp_blackwell_%s//:license" % repo: repo for repo in BLACKWELL_SOURCES},
        mhc_runtime = _KERNELS + "mhc_runtime.h",
        mqa_adapter = "@rtp_blackwell_kernels//:rtp_kernel/fp8_mqa.py",
        sparse_sources = [_KERNELS + f for f in ["sparse_fp8_decode.cpp", "sparse_fp8_kernel.cu"]],
        sparse_headers = [_KERNELS + f for f in ["sparse_fp8_decode.h", "sparse_fp8_layout.h"]],
        elfutils_headers = ["@rtp_blackwell_elfutils//:" + f for f in [
            "libdwfl/libdwfl.h",
            "libdw/libdw.h",
            "libelf/gelf.h",
            "libelf/libelf.h",
        ]],
        tags = ["manual"],
    )
