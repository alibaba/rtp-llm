package(default_visibility = ["//visibility:public"])

[
    sh_test(
        name = name,
        srcs = ["run.sh"],
        args = [mode],
        data = ["lease_role.py", "//rtp_llm/models_py:v41_ppu_shared_sources"],
        env = {"GPU_COUNT": count},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, mode, count in [
        ("components_test", "components", "1"),
        ("native_role_test", "role", "0"),
        ("native_moe_overlap_test", "native_moe", "8"),
    ]
]
