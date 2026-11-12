package(default_visibility = ["//visibility:public"])

py_library(
    name = "scr_checkpoint_probe",
    srcs = ["scr_checkpoint_probe.py"],
    deps = ["//rtp_llm:psutil"],
)

py_library(
    name = "scr_checkpoint_worker",
    srcs = ["scr_checkpoint_worker.py"],
    deps = [":scr_checkpoint_probe"],
)

py_test(
    name = "scr_checkpoint_probe_test",
    srcs = ["scr_checkpoint_probe_test.py"],
    env = {"GPU_COUNT": "0", "PYTHONNOUSERSITE": "1"},
    deps = [":scr_checkpoint_probe"],
)

cc_binary(
    name = "libscr_pinned_probe.so",
    srcs = ["scr_pinned_probe.cc"],
    linkshared = True,
    linkopts = ["-ldl", "-pthread"],
    deps = ["@local_config_cuda//cuda:cupti_headers"],
)

py_test(
    name = "scr_pinned_probe_test",
    srcs = ["scr_pinned_probe_test.py"],
    data = [
        ":libscr_pinned_probe.so",
        "@local_config_cuda//cuda:cupti_dsos",
        "@local_config_cuda//cuda:cudart",
    ],
    env = {"GPU_COUNT": "1", "PYTHONNOUSERSITE": "1"},
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "1"},
    tags = ["H20_CU13", "manual"],
    deps = [":scr_checkpoint_probe"],
)

py_test(
    name = "scr_checkpoint_steady_state_test",
    srcs = ["scr_checkpoint_steady_state_test.py"],
    data = [
        ":libscr_pinned_probe.so",
        "@local_config_cuda//cuda:cupti_dsos",
    ],
    env = {
        "GPU_COUNT": "4",
        "PYTHONNOUSERSITE": "1",
        "SMOKE_ARGS": "--tp_size 4 --dp_size 1 --world_size 4 --act_type FP16 --warm_up 0 --max_seq_len 1024 --kv_cache_mem_mb 64",
        "WORLD_SIZE": "4",
    },
    exec_properties = {"gpu": "H20_CU13", "gpu_count": "4"},
    legacy_create_init = 0,
    tags = ["H20_CU13", "manual", "scr_checkpoint", "smoke_case"],
    timeout = "eternal",
    deps = [
        ":scr_checkpoint_worker",
        "//rtp_llm:requests",
        "//rtp_llm:sdk",
        "//rtp_llm/test/utils:maga_server_manager",
        "//rtp_llm/test/utils:test_util",
    ],
)
