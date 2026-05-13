# Wraps the @remote_kv_cache_manager_client_rpm rpm (declared as http_file in
# `internal_source/deps/http.bzl`) into a cc_library
# `//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client`.
#
# Consumed UNCONDITIONALLY (no arch_select) by
# `//rtp_llm/cpp/cache/connector/remote_connector:client` (see
# `rtp_llm/cpp/cache/connector/remote_connector/BUILD:46-66`), which itself
# is on the dep-graph of `//:th_transformer` for configured accelerator
# builds — removing this BUILD breaks Bazel analysis with
# `no such package '3rdparty/remote_kv_cache_manager'`.
#
# Do NOT delete during deps cleanup unless `remote_connector:client` is also
# being removed and every transitive consumer is updated.
load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "remote_kv_cache_manager_client",
    hdrs = [
        "common.h",
        "manager_client.h",
        "meta_client.h",
        "rtp_llm_client.h",
        "transfer_client.h",
    ],
    include_path = "include/kv_cache_manager",
    include_prefix = "kvcm_client",
    lib_path = "lib",
    rpms = [
        "@remote_kv_cache_manager_client_rpm//file:file",
    ],
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
)
