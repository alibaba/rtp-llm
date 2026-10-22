def _generate_grpc_proto_impl(ctx):
    # input proto file
    proto_file = ctx.file.proto

    # output proto file
    pb2_file = ctx.actions.declare_file(proto_file.basename.replace(".proto", "_pb2.py"))
    pb2_grpc_file = ctx.actions.declare_file(proto_file.basename.replace(".proto", "_pb2_grpc.py"))

    # output dir
    output_dir = ctx.bin_dir.path

    print("create_grpc_proto path: {}".format(ctx.executable.create_grpc_proto.path))
    print("pb2_file path: {}".format(pb2_file))
    print("pb2_grpc_file path: {}".format(pb2_grpc_file))
    print("output_dir: {}".format(output_dir))

    # use create_grpc_proto generate proto py files
    # Preserve the exec-configured py_binary runfiles and the configured action Python.
    tool = ctx.attr.create_grpc_proto[DefaultInfo].files_to_run
    ctx.actions.run_shell(
        outputs = [pb2_file, pb2_grpc_file],
        inputs = [proto_file],
        tools = [tool],
        arguments = [tool.executable.path, proto_file.path, output_dir],
        use_default_shell_env = True,
        command = """set -eu
: "${PYTHON_BIN_PATH:?PYTHON_BIN_PATH must identify the configured execution Python}"
case "$PYTHON_BIN_PATH" in
    */*) export PATH="${PYTHON_BIN_PATH%/*}:${PATH:-/usr/bin:/bin}" ;;
esac
exec "$PYTHON_BIN_PATH" "$1" "$2" "$3"
""",
    )

    # pack as py_library
    return [
        DefaultInfo(files = depset([pb2_file, pb2_grpc_file])),
        PyInfo(
            transitive_sources = depset([pb2_file, pb2_grpc_file]),
        ),
    ]

generate_grpc_proto = rule(
    implementation = _generate_grpc_proto_impl,
    attrs = {
        "proto": attr.label(
            allow_single_file = [".proto"],
            mandatory = True,
        ),
        "create_grpc_proto": attr.label(
            executable = True,
            cfg = "exec",
            mandatory = True,
        ),
    },
)