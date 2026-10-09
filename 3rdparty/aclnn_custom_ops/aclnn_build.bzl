"""AscendC custom-op build rule.

Kernel binary file names embed content hashes, so the opp/ tree cannot be
declared as individual genrule outs (a remote-cache hit would restore an
incomplete package). The rule therefore declares one directory artifact for
opp/ plus regular file outputs for the fixed-name host .so under lib/.
"""

def _aclnn_custom_ops_build_impl(ctx):
    outs = [ctx.actions.declare_file("lib/" + name) for name in ctx.attr.host_libs]
    out_opp = ctx.actions.declare_directory("opp")
    outs.append(out_opp)
    ctx.actions.run_shell(
        inputs = ctx.files.srcs,
        tools = [ctx.file.script],
        outputs = outs,
        command = "bash {script} {out_dir} '{ops}' {soc}".format(
            script = ctx.file.script.path,
            out_dir = out_opp.dirname,
            ops = ctx.attr.op_names,
            soc = ctx.attr.soc_version,
        ),
        mnemonic = "AclnnCustomOpsBuild",
        progress_message = "Building AscendC custom ops ({})".format(ctx.attr.soc_version),
        use_default_shell_env = True,
    )
    return DefaultInfo(files = depset(outs))

aclnn_custom_ops_build = rule(
    implementation = _aclnn_custom_ops_build_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "script": attr.label(allow_single_file = True, mandatory = True),
        "host_libs": attr.string_list(default = [
            "libcust_opapi.so",
            "libcust_opsproto_rt2.0.so",
            "libcust_opmaster_rt2.0.so",
        ]),
        "op_names": attr.string(default = "ALL"),
        "soc_version": attr.string(mandatory = True),
    },
)
