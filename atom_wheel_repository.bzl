"""Pin and patch the ATOM wheel during Bazel repository loading."""

def _patched_atom_wheel_impl(ctx):
    source = ctx.path("original.whl")
    output = ctx.path(ctx.attr.wheel_name)
    script = ctx.path(Label("//patches/atom:repack_wheel.py"))
    patches = [ctx.path(patch) for patch in ctx.attr.patches]
    ctx.download(url = ctx.attr.url, output = source, sha256 = ctx.attr.sha256)
    result = ctx.execute(["python3", script, source, output] + patches, timeout = 600)
    if result.return_code != 0:
        fail("ATOM wheel patch failed: %s%s" % (result.stdout, result.stderr))
    ctx.file("BUILD.bazel", "exports_files([\"%s\"], visibility = [\"//visibility:public\"])\n" % ctx.attr.wheel_name)

patched_atom_wheel = repository_rule(
    implementation = _patched_atom_wheel_impl,
    attrs = {
        "url": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
        "patches": attr.label_list(mandatory = True),
        "wheel_name": attr.string(mandatory = True),
    },
)
