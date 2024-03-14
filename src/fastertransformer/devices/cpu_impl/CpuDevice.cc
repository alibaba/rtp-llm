#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include <cstring>

namespace fastertransformer {

CpuDevice::CpuDevice() {
    allocator_.reset(new Allocator<AllocatorType::CPU>());
}

CpuDevice::~CpuDevice() {
}

void CpuDevice::copy(const CopyParams& params) {
    auto& src = params.src;
    auto& dst = params.dst;
    auto size = params.src.sizeBytes();
    memcpy(dst.data(), src.data(), size);
}


void CpuDevice::layernorm(const LayernormParams& params) {}
void CpuDevice::gemm(const GemmParams& params) {}
void CpuDevice::groupedGemm(const GroupedGemmParams& params) {}
void CpuDevice::embeddingLookup(const EmbeddingLookupParams& params) {}
void CpuDevice::contextAttention(const ContextAttentionParams& params) {}
void CpuDevice::decoderSelfAttention(const AttentionModuleParams& params) {}
void CpuDevice::attentionLayer(const AttentionLayerParams& params) {}
void CpuDevice::ffnLayer(const FfnLayerParams& params) {}
void CpuDevice::sampleTopP(const SamplerParams& params) {}
void CpuDevice::sampleTopK(const SamplerParams& params) {}
void CpuDevice::broadcast(const BroadcastParams& params) {}
void CpuDevice::allReduceSum(const AllReduceParams& params) {}
void CpuDevice::activation(const ActivationParams& params) {}
void CpuDevice::softmax(const SoftmaxParams& params) {}
RTP_LLM_REGISTER_DEVICE(Cpu);


} // namespace fastertransformer
