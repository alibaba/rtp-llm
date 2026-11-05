package(default_visibility = ["//visibility:public"])

exports_files(["test_functional.sh"])

sh_test(
    name = "megamoe_decode_perf_test",
    srcs = ["test_perf.sh"],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "eternal",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "megamoe_perf_contract_test",
    srcs = ["test_perf.sh"],
    args = ["contracts"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual"],
    timeout = "long",
)

sh_test(
    name = "decode_graph_contract_test",
    srcs = ["test_functional.sh"],
    args = ["graph-contracts"],
    env = {"GPU_COUNT": "0", "RTP_V41_TEST_GRAPH_DEVICE": "0"},
    tags = ["manual"],
    timeout = "long",
)

sh_test(
    name = "decode_graph_cache_device_test",
    srcs = ["test_functional.sh"],
    args = ["graph-cache-device"],
    env = {"GPU_COUNT": "1", "RTP_V41_TEST_GRAPH_DEVICE": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "megamoe_decode_graph_test",
    srcs = ["test_perf.sh"],
    args = ["graph"],
    env = {"GPU_COUNT": "8", "RTP_V41_TEST_DECODE_GRAPH": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "eternal",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "decode_graph_fidelity_test",
    srcs = ["test_functional.sh"],
    args = ["graph-fidelity"],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "eternal",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

[
    sh_test(
        name = name,
        srcs = ["test_functional.sh"],
        args = [role],
        env = {"GPU_COUNT": "8"},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, role in [
        ("cp8_prefill_test", "prefill"),
        ("dp8_decode_test", "decode"),
    ]
]
