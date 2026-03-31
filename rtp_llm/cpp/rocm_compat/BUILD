# ROCm compatibility placeholders.
# Empty targets for select() compatibility when ROCm is not available.

config_setting(
    name = "enable_deep_ep",
    define_values = {"enable_deep_ep": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "use_accl_ep",
    define_values = {"use_accl_ep": "true"},
)

cc_library(
    name = "deep_ep_buffer",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_impl",
    visibility = ["//visibility:public"],
)
