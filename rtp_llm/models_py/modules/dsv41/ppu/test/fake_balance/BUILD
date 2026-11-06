package(default_visibility = ["//visibility:public"])

sh_test(
    name = "contracts",
    srcs = ["run.sh"],
    args = ["contracts"],
    env = {"GPU_COUNT": "0"},
    tags = ["manual", "cpu"],
    timeout = "long",
)

sh_test(
    name = "device",
    srcs = ["run.sh"],
    args = ["device"],
    env = {"GPU_COUNT": "8"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
