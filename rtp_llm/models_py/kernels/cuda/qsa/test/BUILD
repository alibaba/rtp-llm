load("//rtp_llm/models_py/test:defs.bzl", "models_py_test")

package(default_visibility = ["//visibility:public"])

models_py_test(
    name = "target_metadata_test",
    src = "target_metadata_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "indexer_paged_score_test",
    src = "indexer_paged_score_test.py",
    gpu = "L20_CU13",
)

models_py_test(
    name = "sparse_fmha_test",
    src = "sparse_fmha_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = ["//rtp_llm/models_py/kernels/cuda/qsa/test/reference:math_references"],
)

models_py_test(
    name = "qsa_tma_prefill_test",
    src = "tma_prefill_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "draft_metadata_test",
    src = "draft_metadata_test.py",
    gpu = "L20_CU13",
)

models_py_test(
    name = "sparse_prefill_bitmap_test",
    src = "sparse_prefill_bitmap_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = ["//rtp_llm/models_py/kernels/cuda/qsa/test/reference:math_references"],
)

models_py_test(
    name = "indexer_prefill_score_test",
    src = "indexer_prefill_score_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "indexer_tensorcore_test",
    src = "indexer_tensorcore_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

models_py_test(
    name = "histogram_topk_test",
    src = "histogram_topk_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
)

models_py_test(
    name = "chunk_paged_gqa_test",
    src = "chunk_paged_gqa_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
    deps = ["//rtp_llm/models_py/kernels/cuda/qsa/test/reference:sparse_paged_reference"],
)

models_py_test(
    name = "grouped_fp8_test",
    src = "grouped_fp8_test.py",
    gpu = "RTX_5000_PRO_CU13",
    testlib = "//rtp_llm/models_py/standalone:auto_model",
    deps = ["//rtp_llm/models_py/kernels/cuda/qsa/test/reference:sparse_paged_reference"],
)

models_py_test(
    name = "indexer_norm_rope_test",
    src = "indexer_norm_rope_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "qsa_packed_count_test",
    src = "qsa_packed_count_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "qsa_cub_topk_test",
    src = "qsa_cub_topk_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "draft_qsa_layout_test",
    src = "draft_qsa_layout_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "draft_paged_fmha_test",
    src = "draft_paged_fmha_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "draft_qsa_metadata_test",
    src = "draft_qsa_metadata_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

py_library(
    testonly = True,
    name = "fp8_kv_test_support",
    srcs = ["fp8_kv_fixture.py"],
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm:testlib"],
)

models_py_test(
    name = "sparse_paged_fmha_test",
    src = "sparse_paged_fmha_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        "//rtp_llm/models_py/kernels/cuda/qsa/test/reference:sparse_paged_reference",
    ],
)

models_py_test(
    name = "fp8_kv_test",
    src = "fp8_kv_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = [
        ":fp8_kv_test_support",
        "//rtp_llm/models_py/kernels/cuda/qsa/test/reference:sparse_paged_reference",
    ],
)

models_py_test(
    name = "topk_test",
    src = "topk_test.py",
    gpu = "L20_CU13",
    deps = [
        "//rtp_llm/models_py/kernels/cuda/qsa/test/reference:math_references",
    ],
)
