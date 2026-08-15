# deps/absent.bzl — explicit absence failure (dependency absence is exhaustive and mutually exclusive).
#
# absent_dep: a rule target that fails at [analysis time] (in the rule implementation),
# referenced by the absence branches of requirement()'s select in arch_config/arch_select.bzl.
# Absence = deps.json exceptions[].exists_in registering "this dependency is unavailable in this
# profile"; when that profile is hit, it must fail explicitly — no falling back to the default
# accelerator (cpu hub) — and point the user to register it in deps.json.
#
# Why fail in the rule implementation rather than at load time: a load-time fail would make
# every config unloadable; select only analyzes the chosen branch, so this fail triggers only
# when the absent profile is actually hit, and other configs (e.g. cuda12_9) load/analyze as usual.

def _absent_dep_impl(ctx):
    fail("Dependency %s is unavailable in profile %s (%s): this is an explicit absence, no fallback to the default accelerator; if needed, register it in deps.json" % (
        ctx.attr.dep_name,
        ctx.attr.profile,
        ctx.attr.reason,
    ))

_absent_dep = rule(
    implementation = _absent_dep_impl,
    attrs = {
        "dep_name": attr.string(mandatory = True),
        "profile": attr.string(mandatory = True),
        "reason": attr.string(default = ""),
    },
)

def absent_dep(name, dep_name, profile, reason = "", visibility = None):
    # tags=["manual"]: the stub may only enter analysis via a select-chosen branch; wildcard
    # //... expansion must skip it, otherwise `bazel build //rtp_llm/...` would treat every
    # profile's absence stubs as build targets and necessarily fail.
    _absent_dep(
        name = name,
        dep_name = dep_name,
        profile = profile,
        reason = reason,
        visibility = visibility,
        tags = ["manual"],
    )
