def _path_inside_wheel(input_file):
    short_path = input_file.short_path
    if short_path.startswith("..") and len(short_path) >= 3:
        separator = short_path[2]
        short_path = short_path[3:]
        position = short_path.find(separator)
        short_path = short_path[position + 1:]
    return short_path

def _input_file_arg(input_file):
    return "%s;%s" % (_path_inside_wheel(input_file), input_file.path)

def _python310_py_wheel_impl(ctx):
    output = ctx.actions.declare_file("-".join([
        ctx.attr.distribution,
        ctx.attr.version,
        ctx.attr.python_tag,
        ctx.attr.abi,
        ctx.attr.platform,
    ]) + ".whl")
    package_inputs = depset(direct = ctx.files.deps)
    input_list = ctx.actions.declare_file(ctx.attr.name + "_inputs.txt")
    input_content = ""
    for input_file in package_inputs.to_list():
        input_content += _input_file_arg(input_file) + "\n"
    ctx.actions.write(output = input_list, content = input_content)

    action_inputs = [input_list]
    args = ctx.actions.args()
    args.add("--name", ctx.attr.distribution)
    args.add("--version", ctx.attr.version)
    args.add("--python_tag", ctx.attr.python_tag)
    args.add("--python_requires", ctx.attr.python_requires)
    args.add("--abi", ctx.attr.abi)
    args.add("--platform", ctx.attr.platform)
    args.add("--out", output.path)
    args.add_all(ctx.attr.strip_path_prefixes, format_each = "--strip_path_prefix=%s")
    args.add("--input_file_list", input_list)

    if ctx.attr.entry_points:
        lines = []
        for group, entries in sorted(ctx.attr.entry_points.items()):
            if lines:
                lines.append("")
            lines.append("[%s]" % group)
            lines += sorted(entries)
        entry_points_file = ctx.actions.declare_file(
            ctx.attr.name + "_entry_points.txt")
        ctx.actions.write(
            output = entry_points_file,
            content = "\n".join(lines))
        action_inputs.append(entry_points_file)
        args.add("--entry_points_file", entry_points_file)

    ctx.actions.run(
        inputs = depset(direct = action_inputs, transitive = [package_inputs]),
        outputs = [output],
        arguments = [args],
        executable = ctx.executable._wheelmaker,
        progress_message = "Building Python 3.10 wheel",
    )
    return [DefaultInfo(
        files = depset([output]),
        data_runfiles = ctx.runfiles(files = [output]),
    )]

python310_py_wheel = rule(
    implementation = _python310_py_wheel_impl,
    attrs = {
        "abi": attr.string(default = "none"),
        "deps": attr.label_list(),
        "distribution": attr.string(mandatory = True),
        "entry_points": attr.string_list_dict(),
        "platform": attr.string(default = "any"),
        "python_requires": attr.string(default = ""),
        "python_tag": attr.string(default = "py3"),
        "strip_path_prefixes": attr.string_list(default = []),
        "version": attr.string(mandatory = True),
        "_wheelmaker": attr.label(
            executable = True,
            cfg = "host",
            default = "//rtp_llm/flexlb/navi_sched/tools:python310_wheelmaker",
        ),
    },
)
