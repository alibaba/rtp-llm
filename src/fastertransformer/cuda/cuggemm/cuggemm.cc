#include "cuggemm.h"

namespace fastertransformer{

void cuggemm::groupGemm(void**        A,
                        void**        B,
                        void**        C,
                        const int*    m,
                        const int*    n,
                        const int*    k,
                        const float   alpha,
                        const float   beta,
                        const int     count)
{
    if (dtype_ == DataType::TYPE_FP16) {
        half_runner_->gemm((half**)A, (half**)B, (half**)C, m, n, k, alpha, beta, count, stream_);
    } else {
        unreachable();
    }

}

}