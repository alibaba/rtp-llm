#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
//#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/cuda/Dispatch.h"

#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
//#include "src/fastertransformer/kernels/sampling_topp_kernels.h"

using namespace std;

namespace fastertransformer {

template<typename In, typename Out>
void mixedTypeSoftmaxWrapper(const SoftmaxParams& params,
                             const Buffer& output,
                             hipStream_t stream)
{
    std::cout<<"successful!";
    MaskedSoftmaxParam<Out, In> param;
    auto& input = params.input;

    // (batch_size, head_num, q_length, k_length)
    param.attention_score    = output.data<Out>();
    // (batch_size, head_num, q_length, k_length)
    param.qk                 = input->data<In>();
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
void inplaceSoftmaxWrapper(const SoftmaxParams& params,
                           hipStream_t stream) {
    mixedTypeSoftmaxWrapper<In, In>(params, *params.input, stream);
}

template<typename Out>
void floatInputSoftmaxWrapper(const SoftmaxParams& params,
                              const BufferPtr& output,
                              hipStream_t stream) {
    mixedTypeSoftmaxWrapper<float, Out>(params, *output, stream);
}

BufferPtr ROCmDevice::softmax(const SoftmaxParams& params) {
    auto input_type = params.input->type();
    auto output_type = (params.output_t != DataType::TYPE_INVALID) ? params.output_t : input_type;

    auto output = (input_type == output_type)
                ? params.input
                : allocateBuffer({output_type, params.input->shape()});
    if (params.mask) {
        RUNTIME_ASSERT_OP_ARG((!params.bias), "cuda softmax does not support bias with mask");
        // if (input_type == output_type) {
        //     DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        //         input_type,
        //         inplaceSoftmaxWrapper,
        //         params,
        //         stream_
        //     );
        if (input_type == output_type) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(input_type,inplaceSoftmaxWrapper,params,stream_);
        } else {
            RUNTIME_ASSERT_OP_ARG(
                input_type == DataType::TYPE_FP32,
                "cuda softmax currently not mixed type with input type [%d]",
                input_type, output_type);
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                output_type,
                floatInputSoftmaxWrapper,
                params,
                output,
                stream_
            );
        }
    } else {
        std::cout<<"mtgu !!!!!";
        // RUNTIME_ASSERT_OP_ARG(
        //     input_type == output_type,
        //     "cuda softmax does not support mixed type without mask, got [%d] and [%d]",
        //     input_type, output_type);
        // DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        //     input_type,
        //     invokeAddBiasSoftMax,
        //     params.input->data(),
        //     params.bias.value().get().data(),
        //     nullptr,
        //     nullptr,
        //     params.input->shape()[0],
        //     params.input->size() / params.input->shape()[0],
        //     params.input->size() / params.input->shape()[0],
        //     stream_
        // );
    }
    return move(output);
}

} // namespace fastertransformer
