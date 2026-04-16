package(default_visibility = ["//visibility:public"])

# gpu_lock is used by bazel test --run_under for GPU resource serialization
py_binary(
    name = "gpu_lock",
    deps = [],
    main = "device_resource.py",
    srcs = [
        "device_resource.py",
        "jit_sys_path_setup.py",
    ],
)
