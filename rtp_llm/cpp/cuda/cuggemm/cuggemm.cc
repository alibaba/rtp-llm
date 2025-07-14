#include "cuggemm.h"

namespace rtp_llm {

void cuggemm::groupGemm(void**      A,
                        void**      B,
                        void**      C,
                        const int*  m,
                        const int*  n,
                        const int*  k,
                        const float alpha,
                        const float beta,
                        const int   count) {
    if (dtype_ == DataType::TYPE_FP16) {
        half_runner_->gemm((half**)A, (half**)B, (half**)C, m, n, k, alpha, beta, count, stream_);
    } else if (dtype_ == DataType::TYPE_BF16) {
        bf16_runner_->gemm(
            (__nv_bfloat16**)A, (__nv_bfloat16**)B, (__nv_bfloat16**)C, m, n, k, alpha, beta, count, stream_);
    } else if (dtype_ == DataType::TYPE_FP32) {
        fp32_runner_->gemm((float**)A, (float**)B, (float**)C, m, n, k, alpha, beta, count, stream_);
    } else {
        RTP_LLM_FAIL("other dtype group gemm not support");
    }
}

}  // namespace rtp_llm
