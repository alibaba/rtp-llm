py_test(
    name = "aux_info_assertions_test",
    srcs = ["aux_info_assertions_test.py"],
    imports = ["../.."],
    deps = [
        "//rtp_llm:pydantic",
    ],
    data = [
        "//rtp_llm/test/smoke:smoke_framework_srcs",
    ],
)

py_test(
    name = "pd_writeback_smoke_config_static_test",
    srcs = ["pd_writeback_smoke_config_static_test.py"],
    data = [
        "//rtp_llm/test/smoke:defs.bzl",
        "//rtp_llm/test/smoke:entry.py",
        "//rtp_llm/test/smoke:data/model/qwen3/q_r_h20_pd_writeback_reuse.json",
        "//rtp_llm/test/smoke:data/model/qwen3/q_r_rocm_pd_writeback_reuse.json",
        "//rtp_llm/test/smoke:smoke_framework_srcs",
        "//rtp_llm/test/smoke:suites_h20_oss.bzl",
        "//rtp_llm/test/smoke:suites_rocm_oss.bzl",
    ],
)
