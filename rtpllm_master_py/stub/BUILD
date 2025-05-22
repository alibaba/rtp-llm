package(default_visibility = ["//visibility:public"])

genrule(
    name = "rtpllm_master_so",
    srcs = ["//rtp_llm/cpp/disaggregate/rtpllm_master/entry:rtpllm_master"],
    outs = [
        "librtpllm_master.so",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

filegroup(
    name = "stubs",
    srcs = [
        ":rtpllm_master_so"
    ],
    data = [
        "librtpllm_master.pyi"
    ],
)
