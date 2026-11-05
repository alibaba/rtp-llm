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
