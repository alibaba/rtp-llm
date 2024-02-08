#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

using namespace std;

namespace fastertransformer {

OpStatus CudaDevice::copy(const CopyParams& params) {
    const auto src = params.src;
    const auto dst = params.dst;
    if (src.data() == dst.data()) {
        return OpStatus::OK();
    }
    if (src.size() != dst.size()) {
        return OpStatus(OpErrorType::ERROR_INVALID_ARGS, "src and dst size mismatch");
    }
    cudaMemcpyKind copyType;
    if (src.where() == MemoryType::MEMORY_GPU && dst.where() != MemoryType::MEMORY_GPU) {
        copyType = cudaMemcpyDeviceToHost;
    } else if (src.where() != MemoryType::MEMORY_GPU && dst.where() == MemoryType::MEMORY_GPU) {
        copyType = cudaMemcpyHostToDevice;
    } else if (src.where() == MemoryType::MEMORY_GPU && dst.where() == MemoryType::MEMORY_GPU) {
        copyType = cudaMemcpyDeviceToDevice;
    } else {
        copyType = cudaMemcpyHostToHost;
    }
    cudaMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), copyType, stream_);
    return OpStatus::OK();
}

OpStatus CudaDevice::layernorm(const LayernormParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::gemm(const GemmParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::groupedGemm(const GroupedGemmParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::contextAttention(const AttentionModuleParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::attentionLayer(const AttentionLayerParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::ffnLayer(const FfnLayerParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::sampleTopP(const SamplerParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::sampleTopK(const SamplerParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::broadcast(const BroadcastParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}

OpStatus CudaDevice::allReduceSum(const AllReduceParams& params) {
    return OpStatus(OpErrorType::ERROR_UNIMPLEMENTED);
}


} // namespace fastertransformer

