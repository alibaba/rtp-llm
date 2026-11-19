"""Build the native operators and their JIT inputs in the installed wheel layout."""

load(":repositories.bzl", "BLACKWELL_SOURCES")

_BINDINGS = "//rtp_llm/models_py/bindings/cuda/blackwell:"
_COMPONENTS = {
    "mhc": ["deep_gemm", "deep_jit", "cutlass_mhc", "fmt"],
    "mqa": ["deep_gemm_mqa", "cutlass_mhc", "fmt"],
    "sparse": ["flash_mla", "cutlass_mla"],
}

def _extensions_impl(ctx):
    outputs = {
        "mhc": ctx.actions.declare_directory("mega_mhc_isolated"),
        "mqa": ctx.actions.declare_directory("fp8_mqa_isolated"),
        "sparse": ctx.actions.declare_file("flash_mla_glm53_22.abi3.so"),
    }
    bindings = {
        "mhc": ctx.file._mhc_binding,
        "mqa": ctx.file._mqa_binding,
        "sparse": ctx.file._sparse_binding,
    }
    sources = {name: target.files for target, name in ctx.attr.sources.items()}
    licenses = {name: target.files.to_list()[0] for target, name in ctx.attr.license_files.items()}
    builder = ctx.attr._builder[DefaultInfo].files_to_run
    for part, repositories in _COMPONENTS.items():
        args = ctx.actions.args()
        args.add(part)
        args.add("--output", outputs[part].path)
        args.add("--binding", bindings[part])
        inputs = [bindings[part]]
        for name in repositories:
            args.add("--source", name + "=" + licenses[name].dirname)
        if part == "mhc":
            args.add("--format-header", ctx.file._format)
            inputs.append(ctx.file._format)
            for name, license in licenses.items():
                args.add("--license", name + "=" + license.path)
            inputs.extend(licenses.values())
        elif part == "mqa":
            inputs.append(licenses["deep_gemm_mqa"])
        ctx.actions.run(
            executable = builder,
            arguments = [args],
            inputs = depset(inputs, transitive = [sources[name] for name in repositories]),
            outputs = [outputs[part]],
            mnemonic = "BlackwellExtension",
            progress_message = "Compiling {} for %{{label}}".format(part),
            use_default_shell_env = True,
        )

    assets = outputs.values() + [ctx.file._check]
    manifest = ctx.actions.declare_file("glm53_extensions.json")
    args = ctx.actions.args()
    args.add("manifest")
    args.add("--output", manifest)
    for asset in assets:
        args.add("--asset", asset.path)
    for patch in ctx.files.patches:
        args.add("--patch", patch)
    pins = {}
    for name, src in BLACKWELL_SOURCES.items():
        pin = {"repository": src.repository, "commit": src.commit}
        if hasattr(src, "sha256"):
            pin["sha256"] = src.sha256
        pins[name] = pin
    args.add("--pins", json.encode(pins))
    ctx.actions.run(
        executable = builder,
        arguments = [args],
        inputs = assets + ctx.files.patches,
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
        "patches": attr.label_list(allow_files = [".patch"]),
        # PyTorch's headers, libraries and CUDAExtension must match the target wheel.
        "_builder": attr.label(default = _BINDINGS + "build_extension", executable = True, cfg = "target"),
        "_mhc_binding": attr.label(default = _BINDINGS + "mhc_binding.cpp", allow_single_file = True),
        "_mqa_binding": attr.label(default = _BINDINGS + "mqa_binding.cpp", allow_single_file = True),
        "_sparse_binding": attr.label(default = _BINDINGS + "sparse_mla_registration.cpp", allow_single_file = True),
        "_format": attr.label(default = _BINDINGS + "format", allow_single_file = True),
        "_check": attr.label(default = "//rtp_llm/libs:glm53_extension_check.py", allow_single_file = True),
    },
)

def blackwell_extensions(name):
    _extensions(
        name = name,
        sources = {"@rtp_blackwell_%s//:sources" % repo: repo for repo in BLACKWELL_SOURCES},
        license_files = {"@rtp_blackwell_%s//:license" % repo: repo for repo in BLACKWELL_SOURCES},
        patches = ["//patches/blackwell:" + p for src in BLACKWELL_SOURCES.values() for p in src.patches],
        tags = ["manual"],
    )
