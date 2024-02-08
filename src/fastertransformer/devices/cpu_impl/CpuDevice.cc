#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"

namespace fastertransformer {

CpuDevice::CpuDevice() {
}

CpuDevice::~CpuDevice() {
}

OpStatus CpuDevice::copy(const CopyParams& params) {}
OpStatus CpuDevice::layernorm(const LayernormParams& params) {}
OpStatus CpuDevice::gemm(const GemmParams& params) {}
OpStatus CpuDevice::groupedGemm(const GroupedGemmParams& params) {}
OpStatus CpuDevice::contextAttention(const AttentionModuleParams& params) {}
OpStatus CpuDevice::decoderSelfAttention(const AttentionModuleParams& params) {}
OpStatus CpuDevice::attentionLayer(const AttentionLayerParams& params) {}
OpStatus CpuDevice::ffnLayer(const FfnLayerParams& params) {}
OpStatus CpuDevice::sampleTopP(const SamplerParams& params) {}
OpStatus CpuDevice::sampleTopK(const SamplerParams& params) {}
OpStatus CpuDevice::broadcast(const BroadcastParams& params) {}
OpStatus CpuDevice::allReduceSum(const AllReduceParams& params) {}


} // namespace fastertransformer
