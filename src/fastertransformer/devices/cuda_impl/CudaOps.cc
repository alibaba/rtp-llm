#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"


using namespace std;

namespace fastertransformer {

void CudaDevice::copy(const CopyParams& params) {
    const auto& src = params.src;
    const auto& dst = params.dst;
    RUNTIME_ASSERT_OP_ARG(src.size() == dst.size(),
        "src and dst size mismatch: [%s] vs [%s]", src.debugString().c_str(), dst.debugString().c_str());

    if (src.data() == dst.data()) {
        return;
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
    std::cout << "copy " << src.sizeBytes() << " bytes to " << dst.data() << std::endl;
    cudaMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), copyType, stream_);
}

void CudaDevice::layernorm(const LayernormParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.input.shape() == params.norm_output.shape(),
        "input and output shape mismatch: [%s] vs [%s]",
        params.input.debugString().c_str(), params.norm_output.debugString().c_str()
    );

    // invokeGeneralLayerNorm(
    //     params.norm_output.data(),
    //     params.input.data(),
    //     params.beta.data(),
    //     params.gamma.data(),
    //     params.eps,

    // );
}

void CudaDevice::groupedGemm(const GroupedGemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}


void CudaDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::attentionLayer(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::sampleTopP(const SamplerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::sampleTopK(const SamplerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::broadcast(const BroadcastParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::allReduceSum(const AllReduceParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}


} // namespace fastertransformer

