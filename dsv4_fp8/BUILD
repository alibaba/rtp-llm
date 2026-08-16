# FP4 -> FP8 routed-expert converter.
#
# Host-only: the converter is numpy/torch tensor algebra plus safetensors header
# rewriting, and the end-to-end cases drive the script by subprocess over a
# synthetic one-shard checkpoint. No GPU and no DeepGEMM, so no tags.
py_test(
    name = "convert_fp4_experts_to_fp8_test",
    srcs = [
        "convert_fp4_experts_to_fp8.py",
        "convert_fp4_experts_to_fp8_test.py",
    ],
    main = "convert_fp4_experts_to_fp8_test.py",
    deps = [
        "//rtp_llm:testlib",
    ],
)
