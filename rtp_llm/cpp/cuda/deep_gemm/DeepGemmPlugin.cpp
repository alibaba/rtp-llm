#include <vector>
#include <map>
#include <algorithm>
#include <torch/torch.h>
#include "rtp_llm/cpp/cuda/deep_gemm/utils.h"
#include "rtp_llm/cpp/cuda/deep_gemm/ConfigUtils.h"
#include "rtp_llm/cpp/core/QBuffer.h"
#include "rtp_llm/cpp/cuda/deep_gemm/DeepGemmPlugin.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/math_utils.h"

using namespace std;

namespace rtp_llm {

#ifdef ENABLE_FP8
void runDeepGemm(__nv_bfloat16* output,
                 __nv_fp8_e4m3* lhs,
                 float*         lhs_scale,
                 __nv_fp8_e4m3* rhs,
                 float*         rhs_scale,
                 int*           grouped_layout,
                 uint32_t       m,
                 uint32_t       n,
                 uint32_t       k,
                 uint32_t       bm,
                 uint32_t       bn,
                 uint32_t       bk,
                 uint32_t       num_groups,
                 uint32_t       num_stages,
                 uint32_t       num_tma_multicast,
                 DeepGemmType   gemm_type,
                 cudaStream_t   stream,
                 uint32_t       num_sms,
                 uint32_t       smem_size,
                 bool           swap_ab);
#endif

size_t DeepGemmPlugin::getPaddingSize(size_t m, DeepGemmType gemm_type) {
    if (gemm_swap_ab_heuristic(m, gemm_type)) {
        // For some reason, m64n8k32 is not used in deep gemm.
        // It might be worth it to activate the shape
        // for some small m in swap ab variant.
        if (m < 16) {
            return 16;
        } else {
            return 8;
        }
    } else {
        return 64;
    }
}

size_t DeepGemmPlugin::paddingMasked(const size_t& token_num) {
    std::vector<size_t> masked_alignment = {16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};

    size_t alignment = masked_alignment.back();
    for (auto& a : masked_alignment) {
        if (token_num <= a) {
            alignment = a;
        }
    }

    return pad(token_num, alignment);
}

inline int DeepGemmPlugin::getNumSms(int user_deep_gemm_num_sm) {
    static int num_sms = -1;
    if (num_sms != -1) {
        return num_sms;
    }
    cudaDeviceProp properties;
    int            device_idx;
    check_cuda_value(cudaGetDevice(&device_idx));
    check_cuda_value(cudaGetDeviceProperties(&properties, device_idx));

    num_sms = properties.multiProcessorCount;

    int num_sms_from_config = user_deep_gemm_num_sm;
    if (num_sms_from_config != -1) {
        num_sms = num_sms_from_config;
    }
    RTP_LLM_LOG_INFO("deep gemm uses sm num %d", num_sms);

    return num_sms;
}

torch::Tensor getColMajorTmaAlignedTensor(Buffer lhs_scale) {
    RTP_LLM_CHECK_WITH_INFO(lhs_scale.dim() == 2 || lhs_scale.dim() == 3, "lhs scale must be dim 2 or 3");
    RTP_LLM_CHECK_WITH_INFO(lhs_scale.type() == DataType::TYPE_FP32, "lhs scale must be fp32");
    int remove_dim = 0;
    if (lhs_scale.dim() == 2) {
        remove_dim = 1;
    }

    size_t g, m, k;
    g = remove_dim ? 1 : lhs_scale.shape()[0];
    m = lhs_scale.shape()[1 - remove_dim];
    k = lhs_scale.shape()[2 - remove_dim];

    int  aligned_m = getTmaAlignedSize(m, lhs_scale.typeSize());
    auto col_major_lhs_scale =
        torch::transpose(torch::empty({int(g), int(k), int(aligned_m)},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)),
                         1,
                         2);
    col_major_lhs_scale.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(0, m), torch::indexing::Slice()},
        torch::from_blob(lhs_scale.data(),
                         {int(g), int(m), int(k)},
                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)));
    if (remove_dim) {
        return col_major_lhs_scale.squeeze(0);
    } else {
        return col_major_lhs_scale;
    }
}

void DeepGemmPlugin::gemmFp8(
    const Buffer& lhs, const Buffer& rhs, Buffer& output, int user_deep_gemm_num_sm, cudaStream_t stream) {
#ifdef ENABLE_FP8
    // lhs.fp8 e4m3, [m, k]; scales -> fp32, [m, k / 128]
    // rhs.fp8 e4m3, [n, k]; scales -> fp32, [n / 128, k / 128]
    // output.bf16, [m, n]
    size_t m, n, k;
    m = lhs.shape()[0];
    k = lhs.shape()[1];
    n = rhs.size() / k;
    RTP_LLM_CHECK_WITH_INFO(n % 64 == 0 && k % 128 == 0, "n(%d) % 64 or k(%d) % 128 != 0", n, k);
    RTP_LLM_LOG_DEBUG("lhs:%s, scale:%s, rhs:%s, scale:%s out:%s",
                      lhs.debugString().c_str(),
                      reinterpret_cast<const QBuffer&>(lhs).scales().debugString().c_str(),
                      rhs.debugString().c_str(),
                      reinterpret_cast<const QBuffer&>(rhs).scales().debugString().c_str(),
                      output.debugString().c_str());
    int  num_sms     = getNumSms(user_deep_gemm_num_sm);
    auto best_config = getBestConfig(m, n, k, 1, num_sms, DeepGemmType::Normal);

    runDeepGemm(output.data<__nv_bfloat16>(),
                reinterpret_cast<const QBuffer&>(lhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(lhs).scales().data<float>(),
                reinterpret_cast<const QBuffer&>(rhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(rhs).scalesData<float>(),
                nullptr,  // grouped_layout
                m,
                n,
                k,
                best_config.block_m,
                best_config.block_n,
                128,  // block_k
                1,    // num_groups
                best_config.num_stages,
                best_config.num_tma_multicast,
                DeepGemmType::Normal,
                stream,
                num_sms,
                best_config.smem_size,
                best_config.swap_ab);
#endif
}

void DeepGemmPlugin::groupedGemmFp8Contiguous(const Buffer& lhs,
                                              const Buffer& rhs,
                                              Buffer&       output,
                                              const Buffer& m_indices,
                                              int           user_deep_gemm_num_sm,
                                              bool          use_64_padding,
                                              cudaStream_t  stream) {
#ifdef ENABLE_FP8
    // lhs.fp8 e4m3, [m_sum, k]; scales -> fp32, [m_sum, k / 128]
    // rhs.fp8 e4m3, [num_groups, n, k]; scales -> fp32, [num_groups, n / 128, k / 128]
    // output.bf16, [m_sum, n]
    // m_indices -> int32, [m_sum]
    size_t m, n, k;
    m              = lhs.shape()[0];
    k              = lhs.shape()[1];
    n              = rhs.shape()[1];
    int num_groups = rhs.shape()[0];
    RTP_LLM_CHECK_WITH_INFO(n % 64 == 0 && k % 128 == 0, "n(%d) % 64 or k(%d) % 128 != 0", n, k);

    auto lhs_scales = getColMajorTmaAlignedTensor(reinterpret_cast<const QBuffer&>(lhs).scales());
    int  num_sms    = getNumSms(user_deep_gemm_num_sm);

    auto best_config = getBestConfig(m, n, k, 1, num_sms, DeepGemmType::GroupedContiguous, -1, use_64_padding);

    runDeepGemm(output.data<__nv_bfloat16>(),
                reinterpret_cast<const QBuffer&>(lhs).kernel().data<__nv_fp8_e4m3>(),
                (float*)lhs_scales.data_ptr(),
                reinterpret_cast<const QBuffer&>(rhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(rhs).scalesData<float>(),
                m_indices.data<int>(),  // grouped_layout
                m,
                n,
                k,
                best_config.block_m,
                best_config.block_n,
                128,         // block_k
                num_groups,  // num_groups
                best_config.num_stages,
                best_config.num_tma_multicast,
                DeepGemmType::GroupedContiguous,
                stream,
                num_sms,
                best_config.smem_size,
                best_config.swap_ab);
#endif
}

void DeepGemmPlugin::groupedGemmFp8Masked(const Buffer& lhs,
                                          const Buffer& rhs,
                                          Buffer&       output,
                                          const Buffer& masked_m,
                                          int           expected_m,
                                          int           user_deep_gemm_num_sm,
                                          cudaStream_t  stream) {
#ifdef ENABLE_FP8
    // lhs.fp8 e4m3, [num_groups, m_max, k]; scales -> fp32, [num_groups, k / 128, m_max]
    // rhs.fp8 e4m3, [num_groups, n, k]; scales -> fp32, [num_groups, n / 128, k / 128]
    // output.bf16, [num_groups, m_max, n]
    // masked_m -> int32, [num_groups]
    size_t m, n, k;
    m              = lhs.shape()[1];
    k              = lhs.shape()[2];
    n              = rhs.shape()[1];
    int num_groups = rhs.shape()[0];
    RTP_LLM_CHECK_WITH_INFO(n % 64 == 0 && k % 128 == 0, "n(%ld) % 64 or k(%ld) % 128 != 0", n, k);

    int num_sms = getNumSms(user_deep_gemm_num_sm);

    auto best_config = getBestConfig(m, n, k, num_groups, num_sms, DeepGemmType::GroupedMasked, expected_m);

    runDeepGemm(output.data<__nv_bfloat16>(),
                reinterpret_cast<const QBuffer&>(lhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(lhs).scalesData<float>(),
                reinterpret_cast<const QBuffer&>(rhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(rhs).scalesData<float>(),
                masked_m.data<int>(),  // grouped_layout
                m,
                n,
                k,
                best_config.block_m,
                best_config.block_n,
                128,         // block_k
                num_groups,  // num_groups
                best_config.num_stages,
                best_config.num_tma_multicast,
                DeepGemmType::GroupedMasked,
                stream,
                num_sms,
                best_config.smem_size,
                best_config.swap_ab);
#endif
}

void DeepGemmPlugin::groupedGemmFp8Masked_V2(const Buffer& lhs,
                                             const Buffer& rhs,
                                             Buffer&       output,
                                             const Buffer& masked_m,
                                             int           expected_m,
                                             int           user_deep_gemm_num_sm,
                                             cudaStream_t  stream) {
#ifdef ENABLE_FP8
    // lhs.fp8 e4m3, [num_groups, m_max, k]; scales -> fp32, [num_groups, m_max, k / 128]
    // rhs.fp8 e4m3, [num_groups, n, k]; scales -> fp32, [num_groups, n / 128, k / 128]
    // output.bf16, [num_groups, m_max, n]
    // masked_m -> int32, [num_groups]
    size_t m, n, k;
    m              = lhs.shape()[1];
    k              = lhs.shape()[2];
    n              = rhs.shape()[1];
    int num_groups = rhs.shape()[0];
    RTP_LLM_CHECK_WITH_INFO(n % 64 == 0 && k % 128 == 0, "n(%ld) % 64 or k(%ld) % 128 != 0", n, k);

    auto lhs_scales = getColMajorTmaAlignedTensor(reinterpret_cast<const QBuffer&>(lhs).scales());
    int  num_sms    = getNumSms(user_deep_gemm_num_sm);

    auto best_config = getBestConfig(m, n, k, num_groups, num_sms, DeepGemmType::GroupedMasked, expected_m);

    runDeepGemm(output.data<__nv_bfloat16>(),
                reinterpret_cast<const QBuffer&>(lhs).kernel().data<__nv_fp8_e4m3>(),
                (float*)lhs_scales.data_ptr(),
                reinterpret_cast<const QBuffer&>(rhs).kernel().data<__nv_fp8_e4m3>(),
                reinterpret_cast<const QBuffer&>(rhs).scalesData<float>(),
                masked_m.data<int>(),  // grouped_layout
                m,
                n,
                k,
                best_config.block_m,
                best_config.block_n,
                128,         // block_k
                num_groups,  // num_groups
                best_config.num_stages,
                best_config.num_tma_multicast,
                DeepGemmType::GroupedMasked,
                stream,
                num_sms,
                best_config.smem_size,
                best_config.swap_ab);
#endif
}

}  // namespace rtp_llm
