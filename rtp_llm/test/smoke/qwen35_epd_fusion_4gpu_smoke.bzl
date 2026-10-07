load("//rtp_llm/test/smoke:defs.bzl", "SMOKE_FRAMEWORK_DEPS")

def qwen35_epd_fusion_4gpu_suite():
    native.py_test(
        name = "qwen35_epd_fusion_4gpu_smoke",
        srcs = ["qwen35_epd_fusion_4gpu_smoke.py"],
        main = "qwen35_epd_fusion_4gpu_smoke.py",
        args = ["--worker"],
        deps = SMOKE_FRAMEWORK_DEPS,
        data = [
            "qwen35_e2p4d2_data/manifest.json",
            "qwen35_e2p4d2_data/messages.json",
            "qwen35_e2p4d2_data/video.mp4",
        ],
        env = {"GPU_COUNT": "4", "WORLD_SIZE": "4", "PYTHONNOUSERSITE": "1", "RTP_GPU_PASSIVE_LOCK": "1", "RTP_GPU_QUERY_TIMEOUT": "60"},
        exec_properties = {"gpu": "L20D", "gpu_count": "4"},
        tags = ["manual", "L20D"],
        timeout = "eternal",
        legacy_create_init = 0,
    )

    native.py_test(
        name = "qwen35_e2pd4_smoke",
        srcs = ["qwen35_epd_fusion_4gpu_smoke.py"],
        main = "qwen35_epd_fusion_4gpu_smoke.py",
        args = ["--worker", "--gpus=4,5,6,7", "--encoder-gpus=0,1"],
        deps = SMOKE_FRAMEWORK_DEPS,
        data = [
            "qwen35_e2p4d2_data/manifest.json",
            "qwen35_e2p4d2_data/messages.json",
            "qwen35_e2p4d2_data/video.mp4",
        ],
        env = {"GPU_COUNT": "6", "WORLD_SIZE": "6", "PYTHONNOUSERSITE": "1", "RTP_GPU_PASSIVE_LOCK": "1", "RTP_GPU_QUERY_TIMEOUT": "60"},
        exec_properties = {"gpu": "L20D", "gpu_count": "6"},
        tags = ["manual", "L20D"],
        timeout = "eternal",
        legacy_create_init = 0,
    )

    native.py_test(
        name = "qwen35_e1pd4_smoke",
        srcs = ["qwen35_epd_fusion_4gpu_smoke.py"],
        main = "qwen35_epd_fusion_4gpu_smoke.py",
        args = ["--worker", "--gpus=4,5,6,7", "--encoder-gpus=0", "--encoder-proxy=1"],
        deps = SMOKE_FRAMEWORK_DEPS,
        data = [
            "qwen35_e2p4d2_data/manifest.json",
            "qwen35_e2p4d2_data/messages.json",
            "qwen35_e2p4d2_data/video.mp4",
        ],
        env = {"GPU_COUNT": "5", "WORLD_SIZE": "5", "PYTHONNOUSERSITE": "1", "RTP_GPU_PASSIVE_LOCK": "1", "RTP_GPU_QUERY_TIMEOUT": "60"},
        exec_properties = {"gpu": "L20D", "gpu_count": "5"},
        tags = ["manual", "L20D"],
        timeout = "eternal",
        legacy_create_init = 0,
    )

    native.py_test(
        name = "qwen35_pd_fusion_4gpu_smoke",
        srcs = ["qwen35_epd_fusion_4gpu_smoke.py"],
        main = "qwen35_epd_fusion_4gpu_smoke.py",
        args = ["--worker", "--workload=text"],
        deps = SMOKE_FRAMEWORK_DEPS,
        data = [
            "qwen35_e2p4d2_data/manifest.json",
            "qwen35_e2p4d2_data/messages.json",
            "qwen35_e2p4d2_data/video.mp4",
        ],
        env = {"GPU_COUNT": "4", "WORLD_SIZE": "4", "PYTHONNOUSERSITE": "1", "RTP_GPU_PASSIVE_LOCK": "1", "RTP_GPU_QUERY_TIMEOUT": "60"},
        exec_properties = {"gpu": "L20D", "gpu_count": "4"},
        tags = ["manual", "L20D"],
        timeout = "eternal",
        legacy_create_init = 0,
    )
