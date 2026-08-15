load(":absent.bzl", "absent_dep")

package(default_visibility = ["//visibility:public"])

absent_dep(
    name = "grpc_benchmark_absent",
    dep_name = "grpc benchmark",
    profile = "all",
    reason = "grpc benchmark-only dependency is not part of RTP-LLM's public graph",
)

absent_dep(
    name = "grpc_gflags_absent",
    dep_name = "gflags",
    profile = "all",
    reason = "grpc test-only dependency is not part of RTP-LLM's public graph",
)

absent_dep(
    name = "grpc_opencensus_stats_absent",
    dep_name = "opencensus-stats",
    profile = "all",
    reason = "grpc optional OpenCensus plugin dependency is not supplied",
)

absent_dep(
    name = "grpc_opencensus_stats_test_absent",
    dep_name = "opencensus-stats-test",
    profile = "all",
    reason = "grpc OpenCensus test dependency is not supplied",
)

absent_dep(
    name = "grpc_opencensus_trace_absent",
    dep_name = "opencensus-trace",
    profile = "all",
    reason = "grpc optional OpenCensus plugin dependency is not supplied",
)

absent_dep(
    name = "grpc_unmapped_absent",
    dep_name = "grpc unmapped external_dep",
    profile = "all",
    reason = "grpc external_deps name is not mapped in _EXTERNAL_DEP_LABELS (e.g. the python-test-only twisted/yaml, or address_sorting); if a C++ target really needs it, map it to a real label in patches/grpc/0002-retire-external-binds.patch",
)
load("@rules_python//python:packaging.bzl", "py_package")
load("@rules_python//python:defs.bzl", "py_binary")
# deps.json must be readable by the extension via Label (cc_view derives the torch/aiter C++ archives).
exports_files(["deps.json"])
# Profile locks must be readable via Label (srcs of the wheel_requires genrule in
# rtp_llm/BUILD; also the source of truth for requirements_lock of pip.parse in MODULE.bazel).
exports_files(glob(["requirements_lock_*.txt"]))

# Generator that derives the aggregated wheel Requires-Dist from deps.json + profile locks.
# stdlib-only, invoked as a tool by the wheel_requires_full / wheel_requires_kserve genrules
# in rtp_llm/BUILD.
py_binary(
    name = "gen_wheel_requires",
    srcs = [
        "compose.py",
        "gen_wheel_requires.py",
        "pkgname.py",
    ],
    main = "gen_wheel_requires.py",
    visibility = ["//visibility:public"],
)

py_package(
    name = "extension_package",
    deps = [
    ],
    packages = [
        "rtp_llm"
    ],
)


py_package(
    name = "extension_package_frontend",
    deps = [
    ],
    packages = [
        "rtp_llm"
    ],
)

py_library(
    name = "extension_frontend_lib",
    deps = [],
    visibility = ["//visibility:public"],
)

py_library(
    name = "extension_lib",
    deps = [],
    visibility = ["//visibility:public"],
)
