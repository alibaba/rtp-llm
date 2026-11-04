# aios/navi_sched/tools/mangled_wheel.bzl
load("//rtp_llm/flexlb/navi_sched/tools:python310_wheel.bzl", "python310_py_wheel")

def _mangle_wheel_impl(ctx):
    output = ctx.actions.declare_file("%s-%s-%s.whl" % (
        ctx.attr.distribution,
        ctx.attr.version,
        ctx.attr.wheel_tag,
    ))
    args = ctx.actions.args()
    args.add("--input", ctx.file.raw_wheel)
    args.add("--output", output)
    args.add("--dist-name", ctx.attr.distribution)
    args.add("--wheel-tag", ctx.attr.wheel_tag)
    ctx.actions.run(
        executable = ctx.executable._mangler,
        inputs = [ctx.file.raw_wheel],
        outputs = [output],
        arguments = [args],
        progress_message = "Finalizing %s wheel" % ctx.attr.distribution,
    )
    return [DefaultInfo(files = depset([output]))]

_mangle_wheel = rule(
    implementation = _mangle_wheel_impl,
    attrs = {
        "distribution": attr.string(mandatory = True),
        "raw_wheel": attr.label(allow_single_file = True, mandatory = True),
        "version": attr.string(mandatory = True),
        "wheel_tag": attr.string(mandatory = True),
        "_mangler": attr.label(
            cfg = "host",
            default = "//rtp_llm/flexlb/navi_sched/tools:mangle_and_flatten",
            executable = True,
        ),
    },
)

def mangled_py_wheel(name, distribution, version, deps, entry_points = {}, strip_path_prefixes = None):
    raw_wheel_name = name + "_raw"
    wheel_tag = select({
        "@platforms//cpu:aarch64": "cp310-cp310-linux_aarch64",
        "//conditions:default": "cp310-cp310-linux_x86_64",
    })

    raw_distribution = distribution + "_raw"

    python310_py_wheel(
        name = raw_wheel_name,
        distribution = raw_distribution,
        version = version,
        deps = deps,
        entry_points = entry_points,
        strip_path_prefixes = strip_path_prefixes,
    )

    _mangle_wheel(
        name = name,
        distribution = distribution,
        raw_wheel = ":" + raw_wheel_name,
        version = version,
        wheel_tag = wheel_tag,
        visibility = ["//visibility:public"],
    )
