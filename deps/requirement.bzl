# deps/requirement.bzl — the single implementation of requirement()'s absence routing.
#
# The open-source @arch_config (arch_config/arch_select.bzl) and the internal overlay
# (internal_source/bazel/arch_select.bzl) each have their own requirement(): the branch tables
# differ, but the invariant "look up absent_map by profile, route to the absent_dep stub on a
# hit, otherwise to the real hub" is one and the same. Each side only provides its own branch
# table; the decision logic exists exactly once.
load("@rules_python//python:defs.bzl", "py_library")
load("//deps:absent_map.bzl", "ABSENT", "ABSENT_REASON")
load("//deps:absent.bzl", "absent_dep")

def norm_dep(name):
    # Must match, character for character, the normalization used to generate absent_map keys
    # (the manifest relock generator: lowercase + both _ and . mapped to hyphens). A mismatch makes
    # the lookup silently miss and the absence branch land on the real hub instead. Loaded by the
    # internal overlay's arch_select.bzl for the same lookup against PRIVATE_ABSENT.
    return name.lower().replace("_", "-").replace(".", "-")

def _branch(name, profile, present, absent_profiles, reason):
    # If a profile registers this dependency as absent in absent_map, that branch does not
    # land on the hub; it points at the analysis-time-failing absent_dep stub (explicit error
    # when the profile is hit, no fallback to the default cpu hub); otherwise use the real hub dependency.
    if profile in absent_profiles:
        stub = name + "__absent__" + profile
        absent_dep(name = stub, dep_name = name, profile = profile, reason = reason)
        return [":" + stub]
    return [present]

def requirement_libs(names, branches, default_profile, default_hub):
    """Create dependency-stub py_library targets per profile.

    branches: [(config_label, profile, hub_requirement_fn)] — accelerator profiles must be
    listed explicitly one by one (a missing branch would wrongly land on the default cpu hub).
    default_profile/default_hub: the //conditions:default branch; only cpu/dev remain.
    """
    for name in names:
        nm = norm_dep(name)
        absent = ABSENT.get(nm, [])
        reason = ABSENT_REASON.get(nm, "")
        deps = {
            config: _branch(name, profile, hub(name), absent, reason)
            for config, profile, hub in branches
        }
        deps["//conditions:default"] = _branch(
            name,
            default_profile,
            default_hub(name),
            absent,
            reason,
        )
        py_library(
            name = name,
            deps = select(deps),
            # tags=["manual"]: dependency stubs enter analysis only via consumers' deps. If
            # wildcard //... expanded them into build targets, the stubs of absent profiles
            # would be analyzed unconditionally and necessarily fail — absence should trigger
            # at a real consumer, not on the wildcard surface.
            tags = ["manual"],
            visibility = ["//visibility:public"],
        )
