package(default_visibility = ["//visibility:public"])

cc_import(
    name = "navi_sched_shared",
    shared_library = "lib/libnavi_sched.so",
)

cc_import(
    name = "python_runtime",
    shared_library = "lib/libpython3.10.so.1.0",
)

[
    cc_import(
        name = name,
        shared_library = "lib/lib%s.so" % name,
    )
    for name in [
        "lz4",
        "zstd",
        "tbb",
        "elf",
        "ev",
    ]
]

cc_library(
    name = "navi_sched_api",
    hdrs = glob(
        ["include/**/*.h"],
        exclude = ["include/rtp_llm/flexlb/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h"],
    ),
    includes = ["include"],
    deps = [
        ":elf",
        ":ev",
        ":lz4",
        ":navi_sched_shared",
        ":python_runtime",
        ":zstd",
    ],
)

cc_library(
    name = "navi_sched",
    hdrs = [
        "include/rtp_llm/flexlb/navi_sched/adapters/rtp_llm/RtpLLMAdapter.h",
    ],
    includes = ["include"],
    deps = [
        ":navi_sched_api",
    ],
)

exports_files([
    "CXX_ABI",
    "VERSION",
] + glob(["lib/*.so"]))
