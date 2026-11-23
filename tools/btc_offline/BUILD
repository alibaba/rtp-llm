load("//bazel:arch_select.bzl", "torch_deps")
load(":offline_target.bzl", "metadata_build_setting", "offline_binary")

package(default_visibility = ["//visibility:public"])

metadata_build_setting(name = "metadata_build", build_setting_default = False)

cc_library(name = "torch_headers_and_runtime", deps = torch_deps())

config_setting(
    name = "without_model_ops",
    flag_values = {":metadata_build": "true"},
)

offline_binary(
    name = "replay",
    binary = "//rtp_llm/cpp/cache/block_tree_cache/benchmark:evaluation_replay",
)

[offline_binary(
    name = t,
    binary = "//rtp_llm/cpp/cache/block_tree_cache/benchmark:" + t,
) for t in ["metadata_pool_test", "metadata_replay_test", "evaluation_replay_test"]]
