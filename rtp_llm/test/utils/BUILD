package(default_visibility = ["//visibility:public"])

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
