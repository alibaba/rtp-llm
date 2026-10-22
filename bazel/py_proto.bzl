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
    ctx.actions.run(
        outputs = [pb2_file, pb2_grpc_file],
        inputs = [proto_file],
        executable = ctx.executable.create_grpc_proto,
        arguments = [proto_file.path, output_dir],
        tools = [ctx.executable.create_grpc_proto]
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