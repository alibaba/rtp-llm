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
    const auto src_offset = params.src_offset;
    const auto dst_offset = params.dst_offset;
    auto copy_length = params.copy_length;

    if (copy_length == 0) {
        RUNTIME_ASSERT_OP_ARG(params.src.shape()[0] == params.dst.shape()[0],
            "src and dst 0d size mismatch: [%s] vs [%s]",
            params.src.debugString().c_str(), params.dst.debugString().c_str());
        copy_length = params.src.shape()[0];
    }

    if (copy_length == 0) {
        return;
    }

    const auto src = params.src.view(src_offset, copy_length);
    const auto dst = params.dst.view(dst_offset, copy_length);

    RUNTIME_ASSERT_OP_ARG(src.sizeBytes() == dst.sizeBytes(),
        "src and dst copy size mismatch: [%s] vs [%s]",
        src.debugString().c_str(), dst.debugString().c_str());

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

    cudaMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), copyType, stream_);
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

