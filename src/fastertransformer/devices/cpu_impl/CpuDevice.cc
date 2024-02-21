#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"

namespace fastertransformer {

CpuDevice::CpuDevice() {
}

CpuDevice::~CpuDevice() {
}

void CpuDevice::copy(const CopyParams& params) {}
void CpuDevice::layernorm(const LayernormParams& params) {}
void CpuDevice::gemm(const GemmParams& params) {}
void CpuDevice::groupedGemm(const GroupedGemmParams& params) {}
void CpuDevice::contextAttention(const AttentionModuleParams& params) {}
void CpuDevice::decoderSelfAttention(const AttentionModuleParams& params) {}
void CpuDevice::attentionLayer(const AttentionLayerParams& params) {}
void CpuDevice::ffnLayer(const FfnLayerParams& params) {}
void CpuDevice::sampleTopP(const SamplerParams& params) {}
void CpuDevice::sampleTopK(const SamplerParams& params) {}
void CpuDevice::broadcast(const BroadcastParams& params) {}
void CpuDevice::allReduceSum(const AllReduceParams& params) {}


} // namespace fastertransformer
