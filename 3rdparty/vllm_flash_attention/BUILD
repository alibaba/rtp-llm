load(
    "@pip_gpu_cuda12_9_torch//:requirements.bzl",
    cuda12_9_requirement = "requirement",
)

_UPSTREAM_ROOT = "upstream/flash_attn/cute/"

filegroup(
    name = "upstream_py_sources",
    srcs = glob([
        _UPSTREAM_ROOT + "**/*.py",
    ]),
    visibility = ["//visibility:public"],
)

_CUTE_PY = [
    "__init__.py",
    "ampere_helpers.py",
    "barrier.py",
    "bench_utils.py",
    "benchmark.py",
    "benchmark_flash_attention_fp8.py",
    "blackwell_helpers.py",
    "block_info.py",
    "block_sparse_utils.py",
    "block_sparsity.py",
    "cache_utils.py",
    "compute_block_sparsity.py",
    "copy_utils.py",
    "cute_dsl_ptxas.py",
    "cute_dsl_utils.py",
    "fa_logging.py",
    "fast_math.py",
    "flash_bwd.py",
    "flash_bwd_mla_dk_sm100.py",
    "flash_bwd_mla_dq_dqv_sm100.py",
    "flash_bwd_mla_sm100.py",
    "flash_bwd_postprocess.py",
    "flash_bwd_preprocess.py",
    "flash_bwd_sm100.py",
    "flash_bwd_sm120.py",
    "flash_bwd_sm90.py",
    "flash_fwd.py",
    "flash_fwd_combine.py",
    "flash_fwd_mla_sm100.py",
    "flash_fwd_sm100.py",
    "flash_fwd_sm120.py",
    "flash_fwd_sm90.py",
    "hopper_helpers.py",
    "interface.py",
    "mask.py",
    "mma_sm100_desc.py",
    "named_barrier.py",
    "pack_gqa.py",
    "paged_kv.py",
    "pipeline.py",
    "seqlen_info.py",
    "sm100_hd256_2cta_fmha_backward.py",
    "sm100_hd256_2cta_fmha_backward_dkdvkernel.py",
    "sm100_hd256_2cta_fmha_backward_dqkernel.py",
    "sm100_hd256_2cta_fmha_forward.py",
    "sm90_config_search.py",
    "softmax.py",
    "testing.py",
    "tile_scheduler.py",
    "topk_gather_kv.py",
    "utils.py",
]

genrule(
    name = "namespace_cute_sources",
    srcs = [":upstream_py_sources"],
    outs = ["vllm_flash_attention/cute/" + src for src in _CUTE_PY],
    cmd = """
set -eu
for src in $(SRCS); do
  rel="$${src#*upstream/flash_attn/cute/}"
  dst="$(RULEDIR)/vllm_flash_attention/cute/$${rel}"
  mkdir -p "$$(dirname "$${dst}")"
  sed 's/flash_attn[.]cute/vllm_flash_attention.cute/g' "$${src}" > "$${dst}"
done
utils="$(RULEDIR)/vllm_flash_attention/cute/utils.py"
test "$$(grep -c '^[[:space:]]*nvvm[.]fmax[(]$$' "$${utils}")" -eq 1
awk '{ print; if ($$0 ~ /^[[:space:]]*nvvm[.]fmax[(]$$/) print "            T.f32()," }' \
  "$${utils}" > "$${utils}.tmp"
mv "$${utils}.tmp" "$${utils}"
""",
)

py_library(
    name = "fa4_cute",
    srcs = [
        "vllm_flash_attention/__init__.py",
        ":namespace_cute_sources",
    ],
    imports = ["."],
    deps = select({
        "@//:using_cuda12_9_x86": [
            cuda12_9_requirement("apache-tvm-ffi"),
            cuda12_9_requirement("einops"),
            cuda12_9_requirement("nvidia-cutlass-dsl"),
            cuda12_9_requirement("quack-kernels"),
            cuda12_9_requirement("torch-c-dlpack-ext"),
            cuda12_9_requirement("torch"),
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)
