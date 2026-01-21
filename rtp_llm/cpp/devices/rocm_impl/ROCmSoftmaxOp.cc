#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/kernels/rocm/layernorm_kernels.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/core/Dispatch.h"

#include "rtp_llm/cpp/kernels/unfused_attention_kernels.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"

using namespace std;

namespace rtp_llm {

template<typename In, typename Out>
void mixedTypeSoftmaxWrapper(const SoftmaxParams& params, const Buffer& output, hipStream_t stream) {
    MaskedSoftmaxParam<Out, In> param;
    auto&                       input = params.input;

    // (batch_size, head_num, q_length, k_length)
    param.attention_score = output.data<Out>();
    // (batch_size, head_num, q_length, k_length)
    param.qk = input->data<In>();
    // (batch_size, q_length, k_length)
    param.attention_mask     = params.mask.value().get().data<Out>();
    param.batch_size         = input->shape()[0];
    param.num_heads          = input->shape()[1];
    param.q_length           = input->shape()[2];
    param.k_length           = input->shape()[3];
    param.qk_scale           = Out(params.scale);
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, stream);
    return;
}

template<typename In>
void inplaceSoftmaxWrapper(const SoftmaxParams& params, hipStream_t stream) {
    mixedTypeSoftmaxWrapper<In, In>(params, *params.input, stream);
}

template<typename Out>
void floatInputSoftmaxWrapper(const SoftmaxParams& params, const BufferPtr& output, hipStream_t stream) {
    mixedTypeSoftmaxWrapper<float, Out>(params, *output, stream);
}

BufferPtr ROCmDevice::softmax(const SoftmaxParams& params) {
    auto input_type  = params.input->type();
    auto output_type = (params.output_t != DataType::TYPE_INVALID) ? params.output_t : input_type;

    auto output = (input_type == output_type) ? params.input : allocateBuffer({output_type, params.input->shape()});
    if (params.mask) {
        RUNTIME_ASSERT_OP_ARG((!params.bias), "cuda softmax does not support bias with mask");
        if (input_type == output_type) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(input_type, inplaceSoftmaxWrapper, params, stream_);
        } else {
            RUNTIME_ASSERT_OP_ARG(input_type == DataType::TYPE_FP32,
                                  "cuda softmax currently not mixed type with input type [%d]",
                                  input_type,
                                  output_type);
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(output_type, floatInputSoftmaxWrapper, params, output, stream_);
        }
    } else {
        RUNTIME_ASSERT_OP_ARG(input_type == output_type,
                              "cuda softmax does not support mixed type without mask, got [%d] and [%d]",
                              input_type,
                              output_type);
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(input_type,
                                         invokeAddBiasSoftMax,
                                         params.input->data(),
                                         params.bias.has_value() ? params.bias.value().get().data() : nullptr,
                                         nullptr,
                                         nullptr,
                                         params.input->shape()[0],
                                         params.input->size() / params.input->shape()[0],
                                         params.input->size() / params.input->shape()[0],
                                         stream_);
    }
    return move(output);
}

}  // namespace rtp_llm
