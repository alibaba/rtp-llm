load("@arch_config//:arch_select.bzl", "copy_all_so")
load("//bazel:defs.bzl", "copy_files", "runfiles_alias")

package(default_visibility = ["//rtp_llm:__subpackages__"])

copy_all_so()

runfiles_alias(
    name = "cache_config_creator_test_runfile",
    src = "//rtp_llm/cpp/cache/test:cache_config_creator_py_test",
    runfile_path = "rtp_llm/libs/test/libcache_config_creator_py_test.so",
)

runfiles_alias(
    name = "context_parallel_test_runfile",
    src = "//rtp_llm/cpp/models/context_parallel/test:th_context_parallel_py_wrapper_test",
    runfile_path = "rtp_llm/libs/test/libth_context_parallel_py_wrapper_test.so",
)

runfiles_alias(
    name = "eplb_test_runfile",
    src = "//rtp_llm/cpp/models/eplb/test:th_eplb_py_wrapper_test",
    runfile_path = "rtp_llm/libs/test/libth_eplb_py_wrapper_test.so",
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
        ":copy_nvcomp_lib",
        ":copy_hf3fs_libs",
        ":copy_remote_kv_cache_manager_client_libs",
        ":libmm_rdma_exporter_so",
        ":librtp_compute_ops_so",
    ],
)

filegroup(
    name = "whl_package_libs",
    srcs = [],
    data = [
        ":copy_nvcomp_lib",
        ":copy_hf3fs_libs",
        ":copy_remote_kv_cache_manager_client_libs",
        ":libmm_rdma_exporter_so",
        ":librtp_compute_ops_so",
        ":libth_grammar_tokenizer_info_so",
        ":libth_transformer_config_so",
        ":libth_transformer_so",
    ],
)
