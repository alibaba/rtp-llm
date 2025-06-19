#pragma once

#include <cuda_runtime_api.h>
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/deep_gemm/utils.h"

namespace rtp_llm {

class DeepGemmPlugin
{
public:
    static size_t getPaddingSize(size_t m, DeepGemmType gemm_type);

    static void gemmFp8(const Buffer &lhs, const Buffer &rhs, Buffer &output, cudaStream_t stream);
    static void groupedGemmFp8Contiguous(const Buffer &lhs, const Buffer &rhs, Buffer &output, const Buffer &m_indices, cudaStream_t stream);
    static void groupedGemmFp8Masked(const Buffer &lhs, const Buffer &rhs, Buffer &output, const Buffer &masked_m, int expected_m, cudaStream_t stream);
    static int user_deep_gemm_num_sm;
private:
    static inline int getNumSms();
};

}
