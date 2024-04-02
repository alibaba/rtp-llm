#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"

#include "src/fastertransformer/kernels/unfused_attention_kernels.h"


using namespace std;

namespace fastertransformer {

template<typename In, typename Out>
BufferPtr softmaxDispatch(const SoftmaxParams& params, CudaDevice* self) {
    const auto& input = params.input;
    MaskedSoftmaxParam<Out, In> param;
    if constexpr (std::is_same<In, Out>::value) {
        // inplace ops
        // (batch_size, head_num, q_length, k_length)
        param.attention_score    = reinterpret_cast<Out*>(params.input.data());
        // (batch_size, head_num, q_length, k_length)
        param.qk                 = reinterpret_cast<In*>(params.input.data());
        // (batch_size, q_length, k_length)
        param.attention_mask     = reinterpret_cast<Out*>(params.mask.data());
        param.batch_size         = params.input.shape()[0];
        param.num_heads          = params.input.shape()[1];
        param.q_length           = params.input.shape()[2];
        param.k_length           = params.input.shape()[3];
        param.qk_scale           = half(params.scale);
        param.linear_bias_slopes = nullptr;
        invokeMaskedSoftmax(param, self->getStream());
        return nullptr;
    } else {
        auto output = self->allocateBuffer({getTensorType<Out>(),
                                            input.shape(),
                                            AllocationType::DEVICE});
        // (batch_size, head_num, q_length, k_length)
        param.attention_score    = reinterpret_cast<Out*>(output->data());
        // (batch_size, head_num, q_length, k_length)
        param.qk                 = reinterpret_cast<In*>(params.input.data());
        // (batch_size, q_length, k_length)
        param.attention_mask     = reinterpret_cast<Out*>(params.mask.data());
        param.batch_size         = params.input.shape()[0];
        param.num_heads          = params.input.shape()[1];
        param.q_length           = params.input.shape()[2];
        param.k_length           = params.input.shape()[3];
        param.qk_scale           = half(params.scale);
        param.linear_bias_slopes = nullptr;
        invokeMaskedSoftmax(param, self->getStream());
        return std::move(output);
    }
    
}


/// @brief   softmax op
BufferPtr CudaDevice::softmax(const SoftmaxParams& params) {
    if (params.input.type() == DataType::TYPE_FP16) {
        if (params.output_t == DataType::TYPE_INVALID) {
            return softmaxDispatch<half, half>(params, this);
        }
    } else if (params.input.type() == DataType::TYPE_FP32) {
        if (params.output_t == DataType::TYPE_INVALID ||
            params.output_t == DataType::TYPE_FP32) {
            return softmaxDispatch<float, float>(params, this);
        } else if (params.output_t == DataType::TYPE_FP16) {
            return softmaxDispatch<float, half>(params, this);
        }
    }
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    
}


} // namespace fastertransformer