load("@arch_config//:arch_select.bzl", "copy_all_so")
load("@rules_python//python:defs.bzl", "py_test")
load("//3rdparty/blackwell:extensions.bzl", "blackwell_extensions")
load("//bazel:defs.bzl", "copy_files", "copy_so")

package(default_visibility = ["//rtp_llm:__subpackages__"])

exports_files(["glm53_extension_check.py"])

blackwell_extensions(name = "blackwell_extensions")

py_test(
    name = "blackwell_abi_test",
    srcs = ["glm53_extension_check.py"],
    main = "glm53_extension_check.py",
    args = ["--allow-cuda-stub"],
    data = [":blackwell_extensions"],
    deps = ["//rtp_llm:numpy", "//rtp_llm:torch"],
    tags = ["manual"],
    target_compatible_with = select({
        "//:using_glm53_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

_SM100_EXTENSIONS = select({
    "//:using_glm53_cuda13_x86": [":blackwell_extensions"],
    "//conditions:default": [],
})

copy_all_so()

copy_so("//rtp_llm/models_py:marlin_nvfp4")

filegroup(
    name = "marlin_nvfp4",
    srcs = select({
        "//:using_cuda13_x86": [":libmarlin_nvfp4_so"],
        "//:using_cuda13_arm": [":libmarlin_nvfp4_so"],
        "//conditions:default": [],
    }),
)

copy_files(
    name = "copy_nvcomp_lib",
    srcs = select({
        "//:using_cuda13_x86": ["@nvcomp_cuda13_x86//:shared_library"],
        "//:using_cuda13_arm": ["@nvcomp_cuda13_arm//:shared_library"],
        "//conditions:default": [],
    }),
)

# copy 3fs libs
copy_files(
    name = "copy_hf3fs_libs",
    srcs = select({
        "//:using_remote_kv_cache": ["//3rdparty/3fs:hf3fs_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

copy_files(
    name = "copy_remote_kv_cache_manager_client_libs",
    srcs = select({
        "//:using_remote_kv_cache": ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_shared"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "frontend_libs",
    srcs = [],
    data = [
        ":libth_transformer_config_so",
    ],
)

filegroup(
    name = "libs",
    srcs = [],
    data = [
        ":marlin_nvfp4",
        ":copy_nvcomp_lib",
        ":copy_hf3fs_libs",
        ":copy_remote_kv_cache_manager_client_libs",
        ":libmm_rdma_exporter_so",
        ":librtp_compute_ops_so",
    ] + _SM100_EXTENSIONS,
)

filegroup(
    name = "whl_package_libs",
    srcs = [],
    data = [
        ":marlin_nvfp4",
        ":copy_nvcomp_lib",
        ":copy_hf3fs_libs",
        ":copy_remote_kv_cache_manager_client_libs",
        ":libmm_rdma_exporter_so",
        ":librtp_compute_ops_so",
        ":libth_grammar_tokenizer_info_so",
        ":libth_transformer_config_so",
        ":libth_transformer_so",
    ] + _SM100_EXTENSIONS,
)
