"""Build the selected native binary without pulling in the model execution graph."""

def _flag_impl(ctx):
    return []

metadata_build_setting = rule(implementation = _flag_impl, build_setting = config.bool(flag = True))

def _offline_impl(settings, attr):
    return {"//tools/btc_offline:metadata_build": True}

_offline = transition(
    implementation = _offline_impl,
    inputs = [],
    outputs = ["//tools/btc_offline:metadata_build"],
)

def _binary_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name)
    binary = ctx.attr.binary[0]
    ctx.actions.symlink(output = output, target_file = binary[DefaultInfo].files_to_run.executable, is_executable = True)
    return [DefaultInfo(executable = output, runfiles = binary[DefaultInfo].default_runfiles)]

offline_binary = rule(
    implementation = _binary_impl,
    executable = True,
    attrs = {
        "binary": attr.label(cfg = _offline, mandatory = True),
        "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
    },
)
