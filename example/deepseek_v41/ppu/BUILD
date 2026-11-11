package(default_visibility = ["//visibility:public"])

exports_files(["test_functional.sh"])

[
    sh_test(
        name = "ptpc_" + name + "_test",
        srcs = ["test_functional.sh"],
        args = ["ptpc-" + name],
        data = [
            "//rtp_llm/models_py:v41_ppu_shared_sources",
            "//rtp_llm/models_py/modules/dsv41/ppu/test:ptpc_sources",
            "//rtp_llm/tools/v41:tooling",
        ],
        env = {"GPU_COUNT": str(gpus)},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, gpus in [("contracts", 0), ("device", 1), ("alignment", 0), ("audit", 0), ("layout", 0)]
]

[
    sh_test(
        name = name,
        srcs = ["test_functional.sh"],
        args = [role],
        data = [
            "//rtp_llm/models_py:v41_ppu_shared_sources",
            "//rtp_llm/models_py/modules/dsv41/ppu/test:shared_eplb_native_sources",
        ],
        env = {
            "GPU_COUNT": "8",
            "RTP_V41_PPU_MOE_BACKEND": "grouped",
            "RTP_V41_PPU_MOE_PRECISION": "w4a8",
        },
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, role in [
        ("shared_eplb_prefill_test", "shared-eplb-prefill"),
        ("shared_eplb_decode_test", "shared-eplb-decode"),
    ]
]

[
    sh_test(
        name = name,
        srcs = ["test_functional.sh"],
        args = [role],
        data = [
            "//rtp_llm/models_py:v41_ppu_shared_sources",
            "//rtp_llm/models_py/modules/dsv41/ppu/test:shared_dspark_modes_native_sources",
        ],
        env = {
            "GPU_COUNT": "8",
            "RTP_V41_PPU_MOE_BACKEND": "grouped",
            "RTP_V41_PPU_MOE_PRECISION": "w4a8",
        },
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, role in [
        ("shared_dspark_modes_prefill_test", "shared-dspark-modes-prefill"),
        ("shared_dspark_modes_decode_test", "shared-dspark-modes-decode"),
    ]
]

[
    sh_test(
        name = name,
        srcs = ["test_functional.sh"],
        args = [role],
        data = [
            "//rtp_llm/models_py:v41_ppu_shared_sources",
            "//rtp_llm/models_py/modules/dsv41/ppu/test:shared_sharded_native_sources",
        ],
        env = {
            "GPU_COUNT": "8",
            "RTP_V41_PPU_MOE_BACKEND": "grouped",
            "RTP_V41_PPU_MOE_PRECISION": "w4a8",
        },
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, role in [
        ("shared_sharded_prefill_test", "shared-sharded-prefill"),
        ("shared_sharded_decode_test", "shared-sharded-decode"),
    ]
]

sh_test(
    name = "shared_optimized_smoke_test",
    srcs = ["test_functional.sh"],
    args = ["shared-optimized"],
    data = [
        "//rtp_llm/models_py:v41_ppu_shared_sources",
        "//rtp_llm/models_py/modules/dsv41/ppu/test:shared_optimized_native_sources",
        "//rtp_llm/models_py/modules/dsv41/ppu/test:shared_graph_native_sources",
    ],
    env = {
        "GPU_COUNT": "8",
        "RTP_V41_PPU_MOE_BACKEND": "megamoe",
        "RTP_V41_PPU_MOE_PRECISION": "w4a4",
    },
    tags = ["manual", "PPU-ZW890P"],
    timeout = "eternal",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "shared_decode_graph_smoke_test",
    srcs = ["test_functional.sh"],
    args = ["shared-graph"],
    data = [
        "//rtp_llm/models_py:v41_ppu_shared_sources",
        "//rtp_llm/models_py/modules/dsv41/ppu/test:shared_graph_native_sources",
    ],
    env = {
        "GPU_COUNT": "8",
        "RTP_V41_PPU_MOE_BACKEND": "megamoe",
        "RTP_V41_PPU_MOE_PRECISION": "w4a4",
    },
    tags = ["manual", "PPU-ZW890P"],
    timeout = "eternal",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "graph_packed_attention_device_test",
    srcs = ["test_functional.sh"],
    args = ["graph-packed-boundary"],
    env = {"GPU_COUNT": "1"},
    tags = ["manual", "PPU-ZW890P"],
    timeout = "long",
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

sh_test(
    name = "shared_grouped_graph_decode_test",
    srcs = ["test_functional.sh"],
    args = ["shared-grouped-graph-decode"],
    data = [
        "//rtp_llm/models_py:v41_ppu_shared_sources",
        "//rtp_llm/models_py/modules/dsv41/ppu/test:shared_graph_modes_native_sources",
    ],
    env = {
        "GPU_COUNT": "8",
        "RTP_V41_PPU_MOE_BACKEND": "grouped",
        "RTP_V41_PPU_MOE_PRECISION": "w4a8",
    },
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
        args = ["shared-dspark-direct-graph", backend, precision] + ([modes] if modes else []),
        data = ["//rtp_llm/models_py/modules/dsv41/ppu/test:shared_dspark_direct_native_sources"],
        env = {
            "GPU_COUNT": "8",
            "RTP_V41_PPU_MOE_BACKEND": backend,
            "RTP_V41_PPU_MOE_PRECISION": precision,
        },
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, backend, precision, modes in [
        ("shared_dspark_direct_graph_test", "grouped", "w4a8", ""),
        ("shared_dspark_megamoe_graph_test", "megamoe", "w4a4", "proposal,commit,dense"),
    ]
]

[
    sh_test(
        name = name,
        srcs = ["test_functional.sh"],
        args = [role],
        data = [
            "//rtp_llm/models_py:v41_ppu_shared_sources",
            "//rtp_llm/models_py/modules/dsv41/ppu/test:shared_native_sources",
        ],
        env = {
            "GPU_COUNT": "8",
            "RTP_V41_PPU_MOE_BACKEND": "grouped",
            "RTP_V41_PPU_MOE_PRECISION": "w4a8",
        },
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, role in [
        ("shared_yoco140_prefill_test", "shared-yoco140-prefill"),
        ("shared_yoco140_decode_test", "shared-yoco140-decode"),
        ("shared_dspark_prefill_test", "shared-dspark-prefill"),
        ("shared_dspark_decode_test", "shared-dspark-decode"),
    ]
]

[
    sh_test(
        name = name,
        srcs = ["test_perf.sh"],
        args = [phase],
        env = {"GPU_COUNT": "8"},
        tags = ["manual", "PPU-ZW890P"],
        timeout = "eternal",
        target_compatible_with = select({
            "@//:using_ppu": [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
    )
    for name, phase in [
        ("cp8_prefill_perf_test", "prefill"),
        ("dp8_decode_perf_test", "decode"),
        ("dsv41_flash_perf_test", "all"),
    ]
]

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
    name = "dp8_graph_attention_ab_test",
    srcs = ["test_perf.sh"],
    args = ["graph-attention-ab"],
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
