load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

package(default_visibility = ["//visibility:public"])

cuda13_sm100_py_test(
    name = "glm5_service_smoke_test",
    main = "service_smoke_test.py",
    srcs = ["service_smoke_test.py"],
    gpu_count = 8,
    tags = ["manual"],
    timeout = "eternal",
    env = {"WORLD_SIZE": "8"},
    deps = [
        ":window_ratio_oracle",
        "//rtp_llm:testlib",
        "//rtp_llm/test/utils:maga_server_manager",
    ],
    data = ["//rtp_llm:sdk"],
)

py_library(
    name = "window_ratio_oracle",
    srcs = ["window_ratio_oracle.py"],
)

py_library(
    name = "expert_load_recorder_test_lib",
    testonly = True,
    srcs = ["expert_load_recorder_test.py"],
    deps = [":window_ratio_oracle", "//rtp_llm/models_py/standalone:py_standalone_testlib"],
)

[
    cuda13_sm100_py_test(
        name = name,
        srcs = [name + ".py"],
        gpu_count = count,
        env = {"GPU_COUNT": str(count)},
        timeout = "long",
        deps = [
            ":window_ratio_oracle",
            "//rtp_llm/models_py/standalone:py_standalone_testlib",
        ] + ([":expert_load_recorder_test_lib"] if count == 2 else []),
    )
    for name, count in [
        ("expert_load_recorder_test", 1),
        ("deepep_router_monitor_test", 1),
        ("expert_load_recorder_ep2_test", 2),
        ("py_model_inputs_eplb_test", 1),
    ]
]
