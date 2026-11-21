py_test(
    name = "cuda_rms_order_test",
    srcs = ["cuda_rms_order_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:torch"],
    tags = ["SM100_ARM_CU13"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_moe_bind_device_test",
    srcs = ["ppu_moe_bind_device_test.py", "ppu_moe_owner_test.py"],
    data = ["//rtp_llm/models_py:ppu_sources"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/models_py/modules/dsv4/ppu/test:ppu_test_runtime",
    ],
    tags = ["PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_moe_owner_test",
    srcs = ["ppu_moe_owner_test.py"],
    data = ["//rtp_llm/models_py:ppu_sources"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
)

py_test(
    name = "moe_owner_test",
    srcs = ["moe_owner_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
)

py_test(
    name = "ppu_attention_owner_test",
    srcs = ["ppu_attention_owner_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
)

py_test(
    name = "ppu_linear_owner_test",
    srcs = ["ppu_linear_owner_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
)

py_test(
    name = "ppu_backbone_layout_test",
    srcs = ["ppu_backbone_layout_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    data = ["//rtp_llm/models_py:ppu_sources"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_whole_backbone_test",
    srcs = [
        "ppu_whole_backbone_test.py",
        "//rtp_llm/models_py:new_models/deepseek_v4/backbone_test.py",
    ],
    main = "ppu_whole_backbone_test.py",
    deps = ["//rtp_llm/models_py:new_weight_loader", "//rtp_llm:models"],
    data = ["//rtp_llm/models_py:ppu_sources"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    env_inherit = ["PPU_FLASH_RELEASE_PATH"],
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_owned_projection_device_test",
    srcs = ["ppu_owned_projection_device_test.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/models_py/modules/dsv4/ppu/test:ppu_test_runtime",
    ],
    data = ["//rtp_llm/models_py:ppu_sources"],
    tags = ["PPU-ZW810E"],
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    target_compatible_with = select({
        "@//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "ppu_globals_test",
    srcs = ["ppu_globals_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
)
