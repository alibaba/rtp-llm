#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/devices/cuda_impl/Dispatch.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"


using namespace std;

namespace fastertransformer {

void CudaDevice::copy(const CopyParams& params) {
    const auto& src = params.src;
    const auto& dst = params.dst;
    const auto& src_offset = params.src_offset;
    const auto& dst_offset = params.dst_offset;
    auto copy_length = params.copy_length ? params.copy_length : min(src.shape()[0], dst.shape()[0]);

    if (copy_length == 0) {
        return;
    }

    RUNTIME_ASSERT_OP_ARG((src.shape()[0] - src_offset >= copy_length),
        "src size is smaller than copy_length: [%s] vs [%d]",
        src.debugString().c_str(), copy_length);
    RUNTIME_ASSERT_OP_ARG((dst.shape()[0] - dst_offset >= copy_length),
        "dst size is smaller than copy_length: [%s] vs [%d]",
        dst.debugString().c_str(), copy_length);

    const auto element_size = src.sizeBytes() / src.shape()[0];
    RUNTIME_ASSERT_OP_ARG((element_size == dst.sizeBytes() / dst.shape()[0]),
        "src and dst element size mismatch: [%s] vs [%s]",
        src.debugString().c_str(), dst.debugString().c_str());

    auto src_ptr = src.data() + src_offset * element_size;
    auto dst_ptr = dst.data() + dst_offset * element_size;
    auto copy_bytes = copy_length * element_size;

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

    cudaMemcpyAsync(dst_ptr, src_ptr, copy_bytes, copyType, stream_);
    sync_check_cuda_error();
}

TransposeOutput CudaDevice::transpose(const TransposeParams& params) {
    const auto& input = params.input;
    const auto data_type = input.type();
    const auto& shape = input.shape();

    RUNTIME_ASSERT_OP_ARG(shape.size() == 2,
        "You can only transpose a 2D buffer, but got [%s]", input.debugString().c_str());

    auto output = allocateBuffer({data_type, {shape[1], shape[0]}});

    DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(data_type, invokeTransposeAxis01,
        output->data(), input.data(), shape[0], shape[1], stream_
    );

    return move(output);
}

GroupedGemmOutput CudaDevice::groupedGemm(const GroupedGemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput CudaDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionLayerOutput CudaDevice::attentionLayer(const AttentionLayerParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::sampleBeamSearch(const BeamSearchParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::broadcast(const BroadcastParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void CudaDevice::allReduceSum(const AllReduceParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}


} // namespace fastertransformer

