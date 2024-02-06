#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

using namespace std;

namespace fastertransformer {

OpStatus CudaDevice::layernorm(LayernormParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::gemm(GemmParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::groupedGemm(GroupedGemmParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::contextAttention(AttentionModuleParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::decoderSelfAttention(AttentionModuleParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::attentionLayer(AttentionLayerParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::ffnLayer(FfnLayerParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::sampleTopP(SamplerParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::sampleTopK(SamplerParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::broadcast(BroadcastParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::allReduceSum(AllReduceParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}


} // namespace fastertransformer

