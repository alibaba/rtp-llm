#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"

namespace fastertransformer {

CpuDevice::CpuDevice() {
}

CpuDevice::~CpuDevice() {
}

void CpuDevice::layernorm(LayernormParams& params) {}
void CpuDevice::gemm(GemmParams& params) {}
void CpuDevice::contextAttention(AttentionModuleParams& params) {}
void CpuDevice::decoderSelfAttention(AttentionModuleParams& params) {}
void CpuDevice::allocateBuffers(AllocateBufferParams& params) {}
void CpuDevice::attentionLayer(AttentionLayerParams& params) {}
void CpuDevice::ffnLayer(FfnLayerParams& params) {}
void CpuDevice::sampleTopP(SamplerParams& params) {}
void CpuDevice::sampleTopK(SamplerParams& params) {}
void CpuDevice::broadcast(BroadcastParams& params) {}
void CpuDevice::allReduceSum(AllReduceParams& params) {}


} // namespace fastertransformer
