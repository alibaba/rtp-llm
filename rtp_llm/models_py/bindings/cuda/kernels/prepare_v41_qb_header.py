"""Build-time adaptation of the locked DeepGEMM device entry, not runtime JIT."""

import pathlib
import re
import sys

source, output = map(pathlib.Path, sys.argv[1:])
text = source.read_text()


def replace(old, new, count=1):
    global text
    if text.count(old) != count:
        raise ValueError(f"DeepGEMM header changed: {old!r}")
    text = text.replace(old, new)


# Consume the existing device-entry patch; never modify the dependency in place.
text, count = re.subn(
    r"#ifdef DG_GEMM_DEVICE_ENTRY\n(.*?)#else.*?#endif",
    lambda match: match.group(1),
    text,
    count=1,
    flags=re.S,
)
if count != 1:
    raise ValueError("Expected the pinned DeepGEMM device-entry patch")
replace("sm100_fp8_fp4_gemm_1d1d_impl(", "v41_qb_gemm_impl(")
# 541b2e2's default ExecutionPolicy still takes a local output descriptor.
argument = (
    "execution_arguments"
    if "const typename ExecutionPolicy::Arguments& execution_arguments) {" in text
    else "tensor_map_cd"
)
replace(f"{argument}) {{", f"{argument}, const QBArgs& qb_args) {{")
replace(
    "namespace deep_gemm {",
    '#include "rtp_llm/models_py/bindings/cuda/kernels/v41_qb_epilogue.cuh"\nnamespace deep_gemm {',
)
loop = "while (scheduler.get_next_block(m_block_idx, n_block_idx)) {"
replace(
    loop,
    loop
    + """
            // Same bijection in all producer/consumer loops: Indexer tiles first.
            n_block_idx = n_block_idx < 4096 / BLOCK_N
                ? n_block_idx + 32768 / BLOCK_N : n_block_idx - 4096 / BLOCK_N;
""",
    count=4,
)
# Insert before dispatch, including the newer optional IndexerQFP4 epilogue.
marker = (
    "            const bool is_empty_group = is_k_grouped_contiguous(kGemmType) "
    "and scheduler.current_shape_k == 0;\n"
)
replace(
    marker,
    marker
    + """
            static_assert(!kSwapAB && BLOCK_M == 128 && BLOCK_N == 128
                          && kNumMulticast == 1 && kNumSMs == 148);
            if (base_n_idx >= 32768) {
                store_indexer_q<BLOCK_M,BLOCK_N,kNumOverlappedTmemCols>(qb_args,tmem_base_addr,
                    base_m_idx,base_n_idx,epilogue_warp_idx,lane_idx,reverse_store_order,
                    &smem.tmem_overlap_barriers[accum_stage_idx],
                    &smem.tmem_empty_barriers[accum_stage_idx]);
                continue;
            }
""",
)
output.write_text(text)
