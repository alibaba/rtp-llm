"""Explicit, environment-tracked selection of independent Hopper repositories."""

def _sm90_dependency_config_impl(ctx):
    value = ctx.os.environ.get("RTP_LLM_SM90_DEPS", "0").lower()
    if value not in ("", "0", "false", "1", "true"):
        fail("RTP_LLM_SM90_DEPS must be unset, 0/false, or 1/true; got %r" % value)
    ctx.file("config.bzl", "SM90_DEPS = %s\n" % ("True" if value in ("1", "true") else "False"))
    ctx.file("BUILD.bazel", 'exports_files(["config.bzl"])\n')
    ctx.file("WORKSPACE", "")

sm90_dependency_config = repository_rule(
    implementation = _sm90_dependency_config_impl,
    environ = ["RTP_LLM_SM90_DEPS"],
    configure = True,
)
