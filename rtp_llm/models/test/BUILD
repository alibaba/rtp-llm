py_test(
    name = "dsv4_kv_cache_test",
    srcs = ["dsv4_kv_cache_test.py"],
    deps = ["//rtp_llm:testlib"],
    exec_properties = {"gpu": "H20"},
)

# Production PD transfer-geometry regression: drives the real
# descs builder + real C++ SpecBuilder via the test-only binding and asserts
# the expected wire geometry per tag/kernel, padding visibility, state/SWA
# kernel independence, layer coverage, and invalid-shape rejection. CPU-only.
py_test(
    name = "pd_transfer_geometry_production_test",
    srcs = ["pd_transfer_geometry_production_test.py"],
    data = [
        "//rtp_llm/cpp/cache/test:cache_config_creator_py_test",
    ],
    deps = ["//rtp_llm:testlib"],
)
