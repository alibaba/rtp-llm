#include "rtp_llm/models_py/bindings/cuda/FusedQKRmsNorm.h"

void FusedQKRMSNorm(at::Tensor&   input,
                    at::Tensor&   q_gamma,
                    at::Tensor&   k_gamma,
                    const double  layernorm_eps,
                    const int64_t q_group_num,
                    const int64_t k_group_num,
                    const int64_t m,
                    const int64_t n,
                    const int64_t norm_size,
                    int64_t       cuda_stream) {
    CHECK_INPUT(input);
    CHECK_INPUT(q_gamma);
    CHECK_INPUT(k_gamma);
    auto device = input.device();
    CHECK_EQ(q_gamma.device(), device);
    CHECK_EQ(k_gamma.device(), device);
    CHECK_DIM(2, input);    // input: (batch_size, hidden_size)
    CHECK_DIM(1, q_gamma);  // weight: (hidden_size)
    CHECK_DIM(1, k_gamma);  // weight: (hidden_size)

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
        rtp_llm::invokeFusedQkRmsNorm(static_cast<c_type*>(input.data_ptr()),
                                      static_cast<c_type*>(q_gamma.data_ptr()),
                                      static_cast<c_type*>(nullptr),
                                      static_cast<c_type*>(k_gamma.data_ptr()),
                                      static_cast<c_type*>(nullptr),
                                      float(layernorm_eps),
                                      q_group_num,
                                      k_group_num,
                                      m,
                                      n,
                                      norm_size,
                                      stream);
        return true;
    });
}