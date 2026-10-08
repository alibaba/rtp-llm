load("//rtp_llm/test/smoke:defs.bzl", "SMOKE_FRAMEWORK_DEPS")

def qwen35_e1pd4_perf_suite():
    # Use the Python launcher: it provides gpu_lock, pinned test_env and paths.
    native.py_test(
        name = "qwen35_e1pd4_perf_smoke",
        srcs = [
            "qwen35_e1pd4_perf_smoke.py",
            "qwen35_epd_fusion_4gpu_smoke.py",
        ],
        main = "qwen35_e1pd4_perf_smoke.py",
        args = ["--worker"],
        deps = SMOKE_FRAMEWORK_DEPS,
        data = [
            "qwen35_e2p4d2_data/manifest.json",
            "qwen35_e2p4d2_data/messages.json",
            "qwen35_e2p4d2_data/video.mp4",
        ],
        env = {
            "GPU_COUNT": "5",
            "WORLD_SIZE": "5",
            "PYTHONNOUSERSITE": "1",
            "RTP_GPU_PASSIVE_LOCK": "1",
            "RTP_GPU_QUERY_TIMEOUT": "60",
        },
        exec_properties = {"gpu": "L20D", "gpu_count": "5"},
        tags = ["manual", "L20D"],
        timeout = "eternal",
        legacy_create_init = 0,
    )
