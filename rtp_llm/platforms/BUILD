# Kept as a filegroup to avoid a dependency cycle: platform adapters use the
# existing model factories, whose runtime data includes these deferred adapters.
filegroup(
    name = "platform_sources",
    srcs = glob(["**/*.py", "**/*.cuh", "**/*.h", "**/*.json", "**/NOTICE*"], exclude = ["**/test/**", "**/tests/**"]),
    visibility = ["//visibility:public"],
)

py_test(
    name = "bootstrap_test",
    srcs = ["ppu/test/test_platform_bootstrap.py"],
    main = "ppu/test/test_platform_bootstrap.py",
    data = [":platform_sources"],
    deps = ["//rtp_llm/models_py:models"],
)

py_test(
    name = "fp8_contract_test",
    srcs = ["ppu/test/test_fp8_contracts.py"],
    main = "ppu/test/test_fp8_contracts.py",
    data = [":platform_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
)

py_test(
    name = "mxfp4_masked_test",
    srcs = ["ppu/test/test_mxfp4_masked.py"],
    main = "ppu/test/test_mxfp4_masked.py",
    data = [":platform_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "decode_hc_test",
    srcs = ["ppu/test/test_decode_hc.py"],
    main = "ppu/test/test_decode_hc.py",
    data = [":platform_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
)

py_test(
    name = "mxfp4_deepep_test",
    srcs = ["ppu/test/test_mxfp4_deepep.py"],
    main = "ppu/test/test_mxfp4_deepep.py",
    data = [":platform_sources"],
    deps = ["//rtp_llm/models_py:models", "//rtp_llm:torch"],
    tags = ["manual", "gpu", "ppu"],
)
