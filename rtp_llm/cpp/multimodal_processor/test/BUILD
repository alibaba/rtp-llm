load("//:def.bzl", "cc_test_wrapper", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [":cuda13_torch_nvshmem"],
    "//conditions:default": [],
})

test_copts = [
    "-fno-access-control",
] + copts()


cc_test(
    name = "multimodal_processor_test",
    srcs = glob([
        "MultimodalProcessorTest.cc",
        "FakeMultimodalProcessor.h",
    ]),
    copts = test_copts,
    deps = torch_deps() + [
        "//rtp_llm/cpp/testing:device_test_utils",

        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//rtp_llm/cpp/multimodal_processor:multimodal_processor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    env = {
        "TEST_USING_DEVICE": "CUDA",
    },
    exec_properties = {'gpu':'H20'},
)

# Ingress ownership / processor-kind decision table. Depends on the lean mm_processor_config
# target rather than the fat one, so it links no torch and needs no GPU node.
cc_test(
    name = "mm_processor_config_test",
    srcs = [
        "MMProcessorConfigTest.cc",
    ],
    copts = test_copts,
    deps = [
        "//rtp_llm/cpp/multimodal_processor:mm_processor_config",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

# LLM-side output transport orchestration: receipt dispatch, the bounded degrade, protocol
# violations and slot-release ordering. Both the control plane and the RDMA transport are fakes.
cc_test(
    name = "mm_remote_output_transport_test",
    srcs = [
        "MMRemoteOutputTransportTest.cc",
    ],
    copts = test_copts,
    deps = torch_deps() + cuda13_torch_link_deps + [
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//rtp_llm/cpp/multimodal_processor/transport:mm_remote_output_transport_core",
        "//rtp_llm/cpp/multimodal_processor/transport/grpc:mm_grpc_transport",
        "//rtp_llm/cpp/multimodal_processor/transport/rdma:mm_rdma_adapter",
        "//rtp_llm/cpp/rdma_transport:rdma_transport",
        # Building inline receipts for the real terminal. Lean target, no engine behind it.
        "//rtp_llm/cpp/model_rpc:tensor_pb_convert",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    # Assertions are CPU-only, but linking torch pulls in libtorch_cuda, which needs
    # libcuda.so.1 just to load the process -- so this still has to land on a GPU node.
    exec_properties = {'gpu':'A10'},
)

# ViT output RDMA slot planning + manifest reassembly, driven by a fake transport and CPU tensors.
cc_test(
    name = "mm_rdma_transport_test",
    srcs = [
        "MMRdmaTransportTest.cc",
    ],
    copts = test_copts,
    deps = torch_deps() + cuda13_torch_link_deps + [
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//rtp_llm/cpp/multimodal_processor/transport/rdma:mm_rdma_adapter",
        "//rtp_llm/cpp/rdma_transport:rdma_transport",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    # Assertions are CPU-only, but linking torch pulls in libtorch_cuda, which needs
    # libcuda.so.1 just to load the process -- so this still has to land on a GPU node.
    exec_properties = {'gpu':'A10'},
)
