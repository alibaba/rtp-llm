"""Build RTP's bindings from the published CUDA extension and DeepGEMM wheels."""

load("@internal_impls//:impls.bzl", "CUDA_EXTENSION", "DEEP_GEMM_WHEEL")
load("@pip_glm53_deepgemm//:requirements.bzl", "requirement")

_BINDINGS = "//rtp_llm/models_py/bindings/cuda/blackwell:"
_MHC_HEADER = "/deep_gemm/csrc/apis/mega_mhc.hpp"

def _find_file(files, matches, description):
    found = [file for file in files if matches(file)]
    if len(found) != 1:
        fail("Expected one {}, found {}".format(description, found))
    return found[0]

def _is_mhc_header(file):
    return file.path.endswith(_MHC_HEADER)

def _is_deep_gemm_license(file):
    return file.basename == "LICENSE" and file.dirname.endswith(".dist-info")

def _extensions_impl(ctx):
    bundle = ctx.attr.source_bundle[DefaultInfo].files
    deep_gemm = ctx.attr.deep_gemm[DefaultInfo].default_runfiles.files
    wheel_files = deep_gemm.to_list()
    mhc_header = _find_file(wheel_files, _is_mhc_header, "DeepGEMM MegaMHC header")
    license_files = ctx.files.deep_gemm_dist_info
    deep_gemm_license = _find_file(license_files, _is_deep_gemm_license, "DeepGEMM wheel license")
    assets = {
        "mhc": ctx.actions.declare_directory("mega_mhc_isolated"),
        "sparse": ctx.actions.declare_file("rtp_fp8_sparse_decode.abi3.so"),
    }
    bindings = {
        "mhc": ctx.file._mhc_binding,
        "sparse": ctx.file._sparse_binding,
    }
    builder = ctx.attr._builder[DefaultInfo].files_to_run
    for part, output in assets.items():
        args = ctx.actions.args()
        args.add(part)
        args.add("--output", output.path)
        args.add("--binding", bindings[part])
        args.add("--bundle", ctx.file.bundle_anchor.dirname)
        inputs = [bindings[part], ctx.file.bundle_anchor]
        sources = [bundle]
        if part == "mhc":
            args.add("--deep-gemm", mhc_header.path[:-len("/csrc/apis/mega_mhc.hpp")])
            args.add("--deep-gemm-license", deep_gemm_license)
            args.add("--format-header", ctx.file._format)
            inputs.extend([deep_gemm_license, ctx.file._format])
            sources.append(deep_gemm)
        ctx.actions.run(
            executable = builder,
            arguments = [args],
            inputs = depset(inputs, transitive = sources),
            outputs = [output],
            mnemonic = "BlackwellExtension",
            progress_message = "Compiling {} for %{{label}}".format(part),
            use_default_shell_env = True,
        )

    adapter = ctx.actions.declare_file("fp8_mqa_adapter.py")
    ctx.actions.symlink(output = adapter, target_file = ctx.file.mqa_adapter)
    files = assets.values() + [ctx.file._check, adapter]
    manifest = ctx.actions.declare_file("glm53_extensions.json")
    args = ctx.actions.args()
    args.add("manifest")
    args.add("--output", manifest)
    for file in files:
        args.add("--asset", file.path)
    args.add("--pins", json.encode({
        "cuda_extension": {
            "commit": CUDA_EXTENSION.commit,
            "sha256": CUDA_EXTENSION.sha256,
        },
        "deep_gemm": {
            "commit": DEEP_GEMM_WHEEL.commit,
            "sha256": DEEP_GEMM_WHEEL.sha256,
        },
    }))
    ctx.actions.run(
        executable = builder,
        arguments = [args],
        inputs = files,
        outputs = [manifest],
        mnemonic = "BlackwellManifest",
        use_default_shell_env = True,
    )
    files.append(manifest)
    return [DefaultInfo(files = depset(files), runfiles = ctx.runfiles(files = files))]

_extensions = rule(
    implementation = _extensions_impl,
    attrs = {
        "source_bundle": attr.label(),
        "bundle_anchor": attr.label(allow_single_file = [".h"]),
        "mqa_adapter": attr.label(allow_single_file = [".py"]),
        "deep_gemm": attr.label(),
        "deep_gemm_dist_info": attr.label(),
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
        source_bundle = "@cuda_extension//:blackwell_sources",
        bundle_anchor = "@cuda_extension//:blackwell/mhc_runtime.h",
        mqa_adapter = "@cuda_extension//:fp8_mqa.py",
        deep_gemm = requirement("deep-gemm"),
        deep_gemm_dist_info = "@pip_glm53_deepgemm_deep_gemm//:dist_info",
        tags = ["manual"],
    )
