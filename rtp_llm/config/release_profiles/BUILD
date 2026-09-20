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
