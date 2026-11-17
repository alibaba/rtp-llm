"""Describe the installed GCC 12 used only by the SM120 grouped binding."""

_DEFAULT_CXX = "/opt/rh/gcc-toolset-12/root/usr/bin/g++"

def _host_tools_impl(ctx):
    compiler = ctx.os.environ.get("RTP_DSV4_HOST_CXX", _DEFAULT_CXX)
    result = ctx.execute([
        ctx.path(ctx.attr._compiler_wrapper),
        "--describe",
        compiler,
    ])
    # Unconfigured/non-x86 queries may load the repository. Only an actual
    # binding build needs the compiler, which checks the identity on execution.
    identity = result.stdout if not result.return_code else compiler + "\n"
    ctx.file("compiler.identity", identity)
    ctx.file("BUILD.bazel", 'exports_files(["compiler.identity"])\n')

dsv4_host_tools = repository_rule(
    implementation = _host_tools_impl,
    local = True,
    configure = True,
    environ = ["RTP_DSV4_HOST_CXX"],
    attrs = {
        "_compiler_wrapper": attr.label(default = Label("//bazel:dsv4_host_compiler.sh")),
    },
)
