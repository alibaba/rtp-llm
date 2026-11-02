py_library(
    name = "tensor_capture_replay",
    srcs = ["tensor_capture_replay.py"],
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "tensor_capture_replay_test",
    size = "small",
    srcs = ["tensor_capture_replay_test.py"],
    args = ["TensorCaptureReplayTest"],
    deps = [
        ":tensor_capture_replay",
        "//rtp_llm:torch",
        "//rtp_llm:utils",
    ],
)

py_test(
    name = "tensor_capture_replay_cuda_test",
    size = "small",
    srcs = ["tensor_capture_replay_test.py"],
    main = "tensor_capture_replay_test.py",
    args = ["TensorCaptureReplayCudaTest"],
    data = ["//rtp_llm/libs:tensor_capture_replay_test_libs"],
    deps = [
        ":tensor_capture_replay",
        "//rtp_llm:ops",
        "//rtp_llm:torch",
        "//rtp_llm:utils",
    ],
    env = {"RTP_LLM_TEST_CAPTURE_REPLAY_CUDA": "1"},
    tags = ["manual", "L20_CU13"],
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
)

py_test(
    name = "tensor_capture_mqa_replay_cuda_test",
    size = "small",
    srcs = ["tensor_capture_replay_test.py"],
    main = "tensor_capture_replay_test.py",
    args = ["TensorCaptureMqaReplayCudaTest"],
    data = ["//rtp_llm/libs:tensor_capture_replay_test_libs"],
    deps = [
        ":tensor_capture_replay",
        "//rtp_llm:ops",
        "//rtp_llm:torch",
        "//rtp_llm:utils",
        "//rtp_llm/models_py:deep_gemm",
    ],
    env = {"RTP_LLM_TEST_CAPTURE_REPLAY_CUDA": "1"},
    tags = ["manual"],
)
