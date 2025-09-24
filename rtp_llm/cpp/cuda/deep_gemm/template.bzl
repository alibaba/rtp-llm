
NORMAL_GEMM_CASES = [
    ("2112", "7168", "1", "DeepGemmType::Normal"),
    ("4096", "7168", "1", "DeepGemmType::Normal"),
    ("7168", "2048", "1", "DeepGemmType::Normal"),
    ("16384", "512", "1", "DeepGemmType::Normal"),
    ("24576", "1536", "1", "DeepGemmType::Normal"),
    ("7168", "16384", "1", "DeepGemmType::Normal"),
    ("36864", "7168", "1", "DeepGemmType::Normal"),
    ("7168", "18432", "1", "DeepGemmType::Normal"),
]

GROUPED_CONTIGUOUS_GEMM_CASES = [
    ("4096", "7168", "256", "DeepGemmType::GroupedContiguous"),
    ("7168", "4096", "256", "DeepGemmType::GroupedContiguous"),
    ("7168", "2048", "256", "DeepGemmType::GroupedContiguous"),
]

GROUPED_MASKED_GEMM_CASES = [
    ("4096", "7168", "256", "DeepGemmType::GroupedMasked"),
    ("7168", "4096", "256", "DeepGemmType::GroupedMasked"),
    ("7168", "2048", "256", "DeepGemmType::GroupedMasked"),
]

dpsk_gemm_so_num = len(NORMAL_GEMM_CASES + GROUPED_CONTIGUOUS_GEMM_CASES + GROUPED_MASKED_GEMM_CASES)

QWEN_NORMAL_CASES = [
    ("9216", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "8192", "1", "DeepGemmType::Normal"),
    ("4608", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "4096", "1", "DeepGemmType::Normal"),
    ("2304", "4096", "1", "DeepGemmType::Normal"),
    ("4096", "2048", "1", "DeepGemmType::Normal"),

    # qwen3-30b-a3b tp=1/2
    # ("5120", "2048", "1", "DeepGemmType::Normal"),
    # ("2048", "4096", "1", "DeepGemmType::Normal"),
    # ("2560", "2048", "1", "DeepGemmType::Normal"),
    # ("2048", "2048", "1", "DeepGemmType::Normal"),
]

QWEN_CONTIGUOUS_CASES = [
    ("3072", "4096", "128", "DeepGemmType::GroupedContiguous"),
    ("4096", "1536", "128", "DeepGemmType::GroupedContiguous"),
]

QWEN_MASKED_CASES = [
    ("3072", "4096", "128", "DeepGemmType::GroupedMasked"),
    ("4096", "1536", "128", "DeepGemmType::GroupedMasked"),
]

qwen_gemm_so_num = len(QWEN_NORMAL_CASES + QWEN_CONTIGUOUS_CASES + QWEN_MASKED_CASES)

template_header = """
#include "rtp_llm/cpp/cuda/deep_gemm/utils.h"
#ifdef ENABLE_FP8
#include "rtp_llm/cpp/cuda/deep_gemm/deep_gemm_template.h"
#include "rtp_llm/cpp/cuda/deep_gemm/include/fp8_gemm.cuh"
#endif
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
namespace rtp_llm {
"""
template = """
#ifdef ENABLE_FP8
template void dispatchBlockNK<{0}, {1}, {2}, {3}>(__nv_bfloat16*         output,
                     __nv_fp8_e4m3*         lhs,
                     float*                 lhs_scale,
                     __nv_fp8_e4m3*         rhs,
                     float*                 rhs_scale,
                     int*                   grouped_layout,
                     uint32_t               m,
                     uint32_t               bm,
                     uint32_t               bn,
                     uint32_t               bk,
                     uint32_t               num_stages,
                     uint32_t               num_tma_multicast,
                     cudaStream_t           stream,
                     uint32_t               num_sms,
                     uint32_t               smem_size,
                     bool                   swap_ab);
#endif
"""
template_tail = """
}
"""


dispatch_template_header = """
#include <cuda.h>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cuda/deep_gemm/utils.h"
#include "rtp_llm/cpp/cuda/deep_gemm/DeepGemmPlugin.h"
#include "rtp_llm/cpp/cuda/deep_gemm/JITRuntime.h"

namespace rtp_llm {
#ifdef ENABLE_FP8
template<uint32_t N, uint32_t K, uint32_t GROUP_NUM, DeepGemmType GEMM_TYPE>
void dispatchBlockNK(__nv_bfloat16*         output,
                     __nv_fp8_e4m3*         lhs,
                     float*                 lhs_scale,
                     __nv_fp8_e4m3*         rhs,
                     float*                 rhs_scale,
                     int*                   grouped_layout,
                     uint32_t               m,
                     uint32_t               bm,
                     uint32_t               bn,
                     uint32_t               bk,
                     uint32_t               num_stages,
                     uint32_t               num_tma_multicast,
                     cudaStream_t           stream,
                     uint32_t               num_sms,
                     uint32_t               smem_size,
                     bool                   swap_ab);

#define DISPATCH_DEEP_GEMM(N, K, GROUP_NUM, GEMM_TYPE)                                           \\
    if (n == N && k == K && num_groups == GROUP_NUM && gemm_type == GEMM_TYPE) {                 \\
        dispatchBlockNK<N, K, GROUP_NUM, GEMM_TYPE>(output,                                      \\
                                                    lhs,                                         \\
                                                    lhs_scale,                                   \\
                                                    rhs,                                         \\
                                                    rhs_scale,                                   \\
                                                    grouped_layout,                              \\
                                                    m,                                           \\
                                                    bm,                                          \\
                                                    bn,                                          \\
                                                    bk,                                          \\
                                                    num_stages,                                  \\
                                                    num_tma_multicast,                           \\
                                                    stream,                                      \\
                                                    num_sms,                                     \\
                                                    smem_size,                                   \\
                                                    swap_ab);                                    \\
        return;                                                                                  \\
    }

void runDeepGemm(__nv_bfloat16*         output,
                 __nv_fp8_e4m3*         lhs,
                 float*                 lhs_scale,
                 __nv_fp8_e4m3*         rhs,
                 float*                 rhs_scale,
                 int*                   grouped_layout,
                 uint32_t               m,
                 uint32_t               n,
                 uint32_t               k,
                 uint32_t               bm,
                 uint32_t               bn,
                 uint32_t               bk,
                 uint32_t               num_groups,
                 uint32_t               num_stages,
                 uint32_t               num_tma_multicast,
                 DeepGemmType           gemm_type,
                 cudaStream_t           stream,
                 uint32_t               num_sms,
                 uint32_t               smem_size,
                 bool                   swap_ab)
{
    RTP_LLM_LOG_DEBUG("m:%u, n:%u, k:%u , bm:%u, bn:%u, bk:%u, num_groups:%u, num_stages:%u, num_tma_multicast:%u, swap_ab:%u\\n", m, n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, swap_ab);
"""

dispatch_template = """
    DISPATCH_DEEP_GEMM({0}, {1}, {2}, {3})
"""

dispatch_template_tail = """
    runKernel(output, lhs, lhs_scale, rhs, rhs_scale, grouped_layout, m, n, k, bm, bn, bk, num_groups, num_stages, num_tma_multicast, gemm_type, stream, num_sms, smem_size, swap_ab);
}
#endif
} // namespace rtp_llm
"""
