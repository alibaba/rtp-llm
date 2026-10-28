config_setting(
    name = "arm_cpu",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "enable_rdma",
    values = {"copt": "-DUSE_RDMA=1"},
)

config_setting(
    name = "enable_solar",
    values = {"copt": "-DACCL_USE_SOLAR=1"},
)

config_setting(
    name = "enable_eic",
    values = {"copt": "-DACCL_USE_EIC=1"},
)

config_setting(
    name = "npu",
    values = {"copt": "-DNPU=1"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "rocm",
    values = {"copt": "-DUSING_ROCM=1"},
    visibility = ["//visibility:public"],
)
