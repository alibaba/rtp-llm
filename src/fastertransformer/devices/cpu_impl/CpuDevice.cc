#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"

namespace fastertransformer {

CpuDevice::CpuDevice() {
}

CpuDevice::~CpuDevice() {
}

OpStatus CpuDevice::layernorm(LayernormParams& params) {}
OpStatus CpuDevice::gemm(GemmParams& params) {}
OpStatus CpuDevice::groupedGemm(GroupedGemmParams& params) {}
OpStatus CpuDevice::contextAttention(AttentionModuleParams& params) {}
OpStatus CpuDevice::decoderSelfAttention(AttentionModuleParams& params) {}
OpStatus CpuDevice::attentionLayer(AttentionLayerParams& params) {}
OpStatus CpuDevice::ffnLayer(FfnLayerParams& params) {}
OpStatus CpuDevice::sampleTopP(SamplerParams& params) {}
OpStatus CpuDevice::sampleTopK(SamplerParams& params) {}
OpStatus CpuDevice::broadcast(BroadcastParams& params) {}
OpStatus CpuDevice::allReduceSum(AllReduceParams& params) {}


} // namespace fastertransformer
