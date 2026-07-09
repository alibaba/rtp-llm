package(default_visibility = ["//visibility:public"])

# Framework source files for cross-package smoke tests.
# Used as data dep (not srcs/deps) to avoid __init__.py creation
# that breaks namespace package merging with internal smoke/.
filegroup(
    name = "smoke_framework_srcs",
    srcs = glob(["*.py"]),
)

exports_files([
    "entry.py",
    "data/prompt_candidates.json",
])

# NOTE: Platform-grouped Bazel test suites were removed during the pyproject/setup.py
# migration.  The framework filegroup and entry/data exports remain here as a thin
# wrapper so internal_source BUILD files that load //rtp_llm/test/smoke:defs.bzl
# and depend on //rtp_llm/test/smoke:smoke_framework_srcs still resolve.
