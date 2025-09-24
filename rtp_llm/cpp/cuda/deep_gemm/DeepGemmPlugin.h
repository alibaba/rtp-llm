#pragma once

#include <cuda_runtime_api.h>
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cuda/deep_gemm/utils.h"

namespace rtp_llm {

class DeepGemmPlugin {
public:
    static size_t getPaddingSize(size_t m, DeepGemmType gemm_type);
    static size_t paddingMasked(const size_t& token_num);

    static void
    gemmFp8(const Buffer& lhs, const Buffer& rhs, Buffer& output, int user_deep_gemm_num_sm, cudaStream_t stream);
    static void groupedGemmFp8Contiguous(const Buffer& lhs,
                                         const Buffer& rhs,
                                         Buffer&       output,
                                         const Buffer& m_indices,
                                         int           user_deep_gemm_num_sm,
                                         cudaStream_t  stream);
    static void groupedGemmFp8Masked(const Buffer& lhs,
                                     const Buffer& rhs,
                                     Buffer&       output,
                                     const Buffer& masked_m,
                                     int           expected_m,
                                     int           user_deep_gemm_num_sm,
                                     cudaStream_t  stream);
    static void groupedGemmFp8Masked_V2(const Buffer& lhs,
                                        const Buffer& rhs,
                                        Buffer&       output,
                                        const Buffer& masked_m,
                                        int           expected_m,
                                        int           user_deep_gemm_num_sm,
                                        cudaStream_t  stream);

private:
    static inline int getNumSms(int user_deep_gemm_num_sm);
};

}  // namespace rtp_llm
