
#include "rtp_llm/models_py/bindings/rocm/Norm.h"
#include "rtp_llm/cpp/cuda/Dispatch.h"
#include "rtp_llm/cpp/core/Types.h"

void layernorm(at::Tensor& output,
               at::Tensor& input,
               at::Tensor& residual,
               at::Tensor& weight,
               at::Tensor& beta,
               double eps,
               int64_t hip_stream) {
    auto device = input.device();

    unsigned int batch_size = input.size(0);
    unsigned int hidden_size = input.size(1);

    
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(DataType::TYPE_FP16, c_type, [&] {
        hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);
        rtp_llm::invokeGeneralLayerNorm(static_cast<c_type*>(nullptr),
            static_cast<c_type*>(output.data_ptr()),
            static_cast<c_type*>(input.data_ptr()),
            static_cast<c_type*>(weight.data_ptr()),
            static_cast<c_type*>(beta.data_ptr()),
            eps, batch_size, hidden_size, stream);
        return true;
    });
}