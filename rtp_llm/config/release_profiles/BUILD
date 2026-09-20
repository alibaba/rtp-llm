# Canonical, runnable deployment settings per release profile.
#
# These files are CONSUMED (by a runner or a deployment) and CHECKED: a unit test fails if a file drifts
# from the profile definition in //rtp_llm/config:release_profile.py. They are the single versioned
# description of what a profile actually runs with, so validation and execution cannot disagree.

exports_files(
    ["sm120_dp4ep4_n2.json"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sm120_dp4ep4_n2",
    srcs = ["sm120_dp4ep4_n2.json"],
    visibility = ["//visibility:public"],
)

# Every profile's settings, packaged with the module that validates them. The glob must live HERE: a
# glob in //rtp_llm cannot cross into this package, so a "config/release_profiles/*.json" glob there
# matches nothing and silently packages an empty set of settings.
filegroup(
    name = "all_profiles",
    srcs = glob(["*.json"]),
    visibility = ["//visibility:public"],
)
