
#include "rtp_llm/models_py/bindings/rocm/Norm.h"
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/core/Types.h"
#include <hip/hip_runtime.h>

namespace rtp_llm {

void layernorm(
    at::Tensor& output, at::Tensor& input, at::Tensor& weight, at::Tensor& beta, double eps, int64_t hip_stream) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    auto device = input.device();
    CHECK_EQ(weight.device(), device);
    CHECK_EQ(beta.device(), device);
    CHECK_DIM(2, input);   // input: (batch_size, hidden_size)
    CHECK_DIM(1, weight);  // weight: (hidden_size)
    CHECK_DIM(1, beta);    // weight: (hidden_size)
    CHECK_EQ(input.size(1), weight.size(0));
    unsigned int batch_size  = input.size(0);
    unsigned int hidden_size = input.size(1);
    CHECK_EQ(output.size(0), batch_size);
    CHECK_EQ(output.size(1), hidden_size);
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);
        invokeGeneralLayerNorm(static_cast<c_type*>(nullptr),
                               static_cast<c_type*>(output.data_ptr()),
                               static_cast<c_type*>(input.data_ptr()),
                               static_cast<c_type*>(weight.data_ptr()),
                               static_cast<c_type*>(beta.data_ptr()),
                               eps,
                               batch_size,
                               hidden_size,
                               stream);
        return true;
    });
}

void fused_add_layernorm(at::Tensor& input,
                         at::Tensor& residual,
                         at::Tensor& bias,
                         at::Tensor& weight,
                         at::Tensor& beta,
                         double      eps,
                         int64_t     hip_stream) {
    CHECK_INPUT(input);
    CHECK_INPUT(residual);
    CHECK_INPUT(bias);
    CHECK_INPUT(weight);
    CHECK_INPUT(beta);
    auto device = input.device();
    CHECK_EQ(residual.device(), device);
    CHECK_EQ(bias.device(), device);
    CHECK_EQ(weight.device(), device);
    CHECK_EQ(beta.device(), device);
    CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
    CHECK_DIM(2, residual);  // input: (batch_size, hidden_size)
    CHECK_DIM(1, bias);      // weight: (hidden_size)
    CHECK_DIM(1, weight);    // weight: (hidden_size)
    CHECK_DIM(1, beta);      // weight: (hidden_size)
    CHECK_EQ(input.size(1), weight.size(0));
    unsigned int batch_size  = input.size(0);
    unsigned int hidden_size = input.size(1);
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);
        invokeGeneralAddBiasResidualLayerNorm(static_cast<c_type*>(residual.data_ptr()),
                                              static_cast<c_type*>(input.data_ptr()),
                                              static_cast<c_type*>(input.data_ptr()),
                                              static_cast<c_type*>(bias.data_ptr()),
                                              static_cast<c_type*>(residual.data_ptr()),
                                              static_cast<c_type*>(weight.data_ptr()),
                                              static_cast<c_type*>(beta.data_ptr()),
                                              eps,
                                              batch_size,
                                              hidden_size,
                                              stream);
        return true;
    });
}

}  // namespace rtp_llm
