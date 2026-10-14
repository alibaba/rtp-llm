def _cutlass_install_impl(ctx):
    output = ctx.actions.declare_directory(ctx.label.name)
    args = ctx.actions.args()
    args.add("--output", output.path)
    args.add("--version", ctx.attr.version)
    args.add_all(ctx.files.wheels)
    ctx.actions.run(
        executable = ctx.attr._merger[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = ctx.files.wheels,
        outputs = [output],
        mnemonic = "AssembleCutlassDSL",
        progress_message = "Assembling CUTLASS DSL %s" % ctx.attr.version,
    )
    return [DefaultInfo(files = depset([output]), runfiles = ctx.runfiles(files = [output]))]

cutlass_install = rule(
    implementation = _cutlass_install_impl,
    attrs = {
        "wheels": attr.label_list(allow_files = [".whl"], mandatory = True),
        "version": attr.string(mandatory = True),
        "_merger": attr.label(
            default = "//bazel:merge_cutlass_wheels",
            executable = True,
            cfg = "exec",
        ),
    },
)
