load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test(
    name = "uqi_graph_without_two_pass_metadata_test",
    srcs = ["embedding_input_metadata_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/embedding_engine:embedding_input_metadata",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    env = {
        "USE_VISION_BERT_UQI_BLOCK_MASK": "1",
        "VISION_BERT_UQI_TWO_PASS": "0",
        "ENABLE_CUDA_GRAPH": "1",
    },
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)

cc_test(
    name = "qi_device_metadata_test",
    srcs = ["embedding_input_metadata_test.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/embedding_engine:embedding_input_metadata",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    env = {
        "USE_VISION_BERT_UQI_BLOCK_MASK": "0",
        "VISION_BERT_UQI_TWO_PASS": "0",
        "ENABLE_CUDA_GRAPH": "1",
    },
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
)
