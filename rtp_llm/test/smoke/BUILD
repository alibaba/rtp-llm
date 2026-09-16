package(default_visibility = ["//visibility:public"])

# Framework source files for cross-package smoke tests.
filegroup(
    name = "smoke_framework_srcs",
    srcs = glob(["*.py"]),
)


exports_files([
    "entry.py",
    "data/prompt_candidates.json",
])

# Platform-grouped Bazel suites were removed during the pyproject/setup.py
# migration. Internal smoke targets still use this thin compatibility wrapper.
