#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"

#include "src/fastertransformer/kernels/unfused_attention_kernels.h"


using namespace std;

namespace fastertransformer {


/// @brief   softmax op
void CudaDevice::softmax(const SoftmaxParams& params) {

    MaskedSoftmaxParam<half, float> param;
    // (batch_size, head_num, q_length, k_length)
    param.attention_score    = reinterpret_cast<half*>(params.output.data());
    // (batch_size, head_num, q_length, k_length)
    param.qk                 = reinterpret_cast<float*>(params.input.data());
    // (batch_size, q_length, k_length)
    param.attention_mask     = reinterpret_cast<half*>(params.mask.data());  
    param.batch_size         = params.input.shape()[0];
    param.num_heads          = params.input.shape()[1];
    param.q_length           = params.input.shape()[2];
    param.k_length           = params.input.shape()[3];
    param.qk_scale           = half(params.scale);
    param.linear_bias_slopes = nullptr;
    invokeMaskedSoftmax(param, stream_);
    
}


} // namespace fastertransformer