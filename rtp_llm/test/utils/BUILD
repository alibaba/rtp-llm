package(default_visibility = ["//visibility:public"])

# test_util provides shared Python test helpers (numeric comparison, port
# allocation, stream utilities, etc.) used by Bazel py_test targets that
# have not yet been migrated to pytest.
py_library(
    name = "test_util",
    srcs = [
        "bench_util.py",
        "numeric_util.py",
        "cuda_graph_util.py",
        "port_util.py",
        "stream_util.py",
        "diff_util.py",
        "coredump_util.py",
    ],
    deps = [],
)

py_library(
    name = "coredump_util",
    srcs = ["coredump_util.py"],
    visibility = ["//visibility:public"],
)

# gpu_lock is a Bazel py_binary used as `--run_under` for cc_test targets
# (see github-opensource/setup.py BazelTest command). It serializes GPU
# access across parallel cpp_ut runs by acquiring a filelock per device
# in /tmp/rtp_llm/smoke/test/gpu_status/, then execs the test as a
# subprocess. Pure stdlib + filelock — no torch import on the wrapper
# path, so no extra Bazel deps are required.
#
# This BUILD file survived the pyproject/uv migration only for this one
# target; everything else under rtp_llm/test/utils/ is exercised via
# pytest, not Bazel.
py_binary(
    name = "gpu_lock",
    srcs = [
        "device_resource.py",
        "jit_sys_path_setup.py",
    ],
    main = "device_resource.py",
)
