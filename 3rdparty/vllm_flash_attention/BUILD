load(
    "@pip_gpu_cuda12_9_torch//:requirements.bzl",
    cuda12_9_requirement = "requirement",
)

_UPSTREAM_ROOT = "upstream/flash_attn/cute/"

# Flat glob: namespace_cute_sources declares outs by basename, so a nested file
# would be generated but never declared, and silently dropped.
# Private: these sources still import `flash_attn.cute` and collide with the
# installed flash-attn 2 wheel. Depend on :fa4_cute.
filegroup(
    name = "upstream_py_sources",
    srcs = glob([
        _UPSTREAM_ROOT + "*.py",
    ]),
    visibility = ["//visibility:private"],
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
declared=%d
actual=$$(set -- $(SRCS); echo $$#)
if [ "$${actual}" -ne "$${declared}" ]; then
  echo "FA4 vendoring drift: upstream/flash_attn/cute has $${actual} .py files, _CUTE_PY declares $${declared}." >&2
  echo "Refresh _CUTE_PY and SHA256SUMS, see 3rdparty/vllm_flash_attention/README.md upgrade flow." >&2
  exit 1
fi
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
""" % len(_CUTE_PY),
)

# BSD-3-Clause binary redistribution must reproduce the copyright notice.
# AUTHORS ships too because LICENSE names its holders by reference to it.
filegroup(
    name = "upstream_license_files",
    srcs = [
        _UPSTREAM_ROOT + "AUTHORS",
        _UPSTREAM_ROOT + "LICENSE",
    ],
    visibility = ["//visibility:private"],
)

genrule(
    name = "package_license_files",
    srcs = [":upstream_license_files"],
    outs = [
        "vllm_flash_attention/AUTHORS",
        "vllm_flash_attention/LICENSE",
    ],
    cmd = """
set -eu
mkdir -p "$(RULEDIR)/vllm_flash_attention"
for src in $(SRCS); do
  cp "$${src}" "$(RULEDIR)/vllm_flash_attention/$$(basename "$${src}")"
done
""",
)

py_library(
    name = "fa4_cute",
    srcs = [
        "vllm_flash_attention/__init__.py",
        ":namespace_cute_sources",
    ],
    data = [":package_license_files"],
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
