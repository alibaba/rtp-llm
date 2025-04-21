#pragma once

#include <cuda_runtime_api.h>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/deep_gemm/utils.h"

namespace fastertransformer {

class DeepGemmPlugin
{
public:
    static void gemmFp8(const Buffer &lhs, const Buffer &rhs, Buffer &output, cudaStream_t stream);
    static void groupedGemmFp8Contiguous(const Buffer &lhs, const Buffer &rhs, Buffer &output, const Buffer &m_indices, cudaStream_t stream);
    static void groupedGemmFp8Masked(const Buffer &lhs, const Buffer &rhs, Buffer &output, const Buffer &masked_m, int expected_m, cudaStream_t stream);
private:
    static inline int getNumSms();
};

}
