#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"

#include "src/fastertransformer/kernels/unfused_attention_kernels.h"


using namespace std;

namespace fastertransformer {

template<typename In>
void inplaceSoftmaxWrapper(const SoftmaxParams& params,
                           cudaStream_t stream) {
    MaskedSoftmaxParam<In, In> param;
    // inplace ops
    // (batch_size, head_num, q_length, k_length)
    param.attention_score    = params.input->data<In>();
    // (batch_size, head_num, q_length, k_length)
    param.qk                 = params.input->data<In>();
    // (batch_size, q_length, k_length)
    param.attention_mask     = params.mask.data<In>();
    param.batch_size         = params.input->shape()[0];
    param.num_heads          = params.input->shape()[1];
    param.q_length           = params.input->shape()[2];
    param.k_length           = params.input->shape()[3];
    param.qk_scale           = half(params.scale);
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, stream);
    return;
}

template<typename In, typename Out>
void mixFloatSoftmaxWrapper(const SoftmaxParams& params,
                            const Buffer& output,
                            cudaStream_t stream) {
    MaskedSoftmaxParam<Out, In> param;
    auto& input = params.input;

    // (batch_size, head_num, q_length, k_length)
    param.attention_score    = output.data<Out>();
    // (batch_size, head_num, q_length, k_length)
    param.qk                 = input->data<In>();
    // (batch_size, q_length, k_length)
    param.attention_mask     = params.mask.data<Out>();
    param.batch_size         = input->shape()[0];
    param.num_heads          = input->shape()[1];
    param.q_length           = input->shape()[2];
    param.k_length           = input->shape()[3];
    param.qk_scale           = half(params.scale);
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, stream);
    return;
}

/// @brief   softmax op
BufferPtr CudaDevice::softmax(const SoftmaxParams& params) {
    if (params.input == nullptr) {
        throw std::runtime_error("softmax input can not be nullptr");
    }
    auto type = params.input->type();
    // inplace
    if (type == params.output_t) {
        if (type == DataType::TYPE_FP16) {
            inplaceSoftmaxWrapper<half>(params, stream_);
        } else if (type == DataType::TYPE_FP32) {
            inplaceSoftmaxWrapper<float>(params, stream_);
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
        return BufferPtr(params.input);
    } else {
        if (type == DataType::TYPE_FP32 && params.output_t == DataType::TYPE_FP16) {
            auto output = allocateBuffer({params.output_t,
                                          params.input->shape(),
                                          AllocationType::DEVICE});
            mixFloatSoftmaxWrapper<float, half>(params, *output, stream_);
            return std::move(output);
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }

    }
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);

}


} // namespace fastertransformer