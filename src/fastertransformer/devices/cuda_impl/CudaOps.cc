#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/devices/cuda_impl/Dispatch.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils_torch.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"


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

template<typename DstT, typename SrcT>
inline DstT castTo(SrcT value) {
    return static_cast<DstT>(value);
}

#define SPECIALIZE_CAST_TO(DstT, SrcT)             \
    template<>                                     \
    inline DstT castTo<DstT, SrcT>(SrcT value) {   \
        return static_cast<DstT>((float)value);    \
    }

SPECIALIZE_CAST_TO(int64_t, __half);
SPECIALIZE_CAST_TO(uint64_t, __half);
SPECIALIZE_CAST_TO(int64_t, __nv_bfloat16);
SPECIALIZE_CAST_TO(uint64_t, __nv_bfloat16);
SPECIALIZE_CAST_TO(__half, int64_t);
SPECIALIZE_CAST_TO(__half, uint64_t);
SPECIALIZE_CAST_TO(__nv_bfloat16, int64_t);
SPECIALIZE_CAST_TO(__nv_bfloat16, uint64_t);
SPECIALIZE_CAST_TO(__nv_bfloat16, __half);
SPECIALIZE_CAST_TO(__half, __nv_bfloat16);

// TODO: change this to use efficient cuda kernel
template<typename DstT, typename SrcT>
void convertType(const void* dst, void* src, size_t size) {
    const auto src_ptr = (const SrcT*)(src);
    auto dst_ptr = (DstT*)(dst);

    for (size_t i = 0; i < size; ++i) {
        dst_ptr[i] = castTo<DstT>(src_ptr[i]);
    }
}

ConvertOutput CudaDevice::convert(const ConvertParams& params) {
    const auto& input = params.input;
    if (input->type() == params.type) {
        return input;
    }

    auto alloc_type = getMemAllocationType(input->where());
    auto host_input = (alloc_type == AllocationType::HOST) ? input : clone({*input, AllocationType::HOST});
    auto host_output = allocateBuffer({params.type, input->shape(), AllocationType::HOST});
    syncAndCheck();
    DISPATCH_CUDA_FUNCTION_TWO_TYPES(host_output->type(), input->type(), convertType,
        host_output->data(), host_input->data(), host_input->size()
    );

    auto output = (alloc_type == AllocationType::HOST) ? host_output : clone({*host_output, alloc_type});
    return {move(output)};
}

SelectOutput CudaDevice::select(const SelectParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.dim == 0, "select op tmp only support dim == 0");
    const auto& input = params.input;
    auto alloc_type = getMemAllocationType(input.where());
    auto shape = input.shape();
    shape[0] = params.index.size();
    auto output = allocateBuffer({input.type(), shape, alloc_type});
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(input.type(), invokeLookupHiddenStateOfLastToken, output->data(), input.data(), (int*)params.index.data(), (int)params.index.size(), (int)shape[1], stream_);
    return {std::move(output)};
}

inline ncclDataType_t getNcclDataType(DataType type) {
    switch (type) {
        case DataType::TYPE_INT8: return ncclInt8;
        case DataType::TYPE_INT32: return ncclInt32;
        case DataType::TYPE_INT64: return ncclInt64;
        case DataType::TYPE_UINT8: return ncclUint8;
        case DataType::TYPE_UINT32: return ncclUint32;
        case DataType::TYPE_UINT64: return ncclUint64;
        case DataType::TYPE_FP16: return ncclFloat16;
        case DataType::TYPE_FP32: return ncclFloat32;
        case DataType::TYPE_FP64: return ncclFloat64;
        case DataType::TYPE_BF16: return ncclBfloat16;
        default: throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

void CudaDevice::broadcast(const BroadcastParams& params) {
    if (nccl_param_.world_size_ < 2) {
        return;
    }

    for (auto i = 0; i < params.buffers.size(); ++i) {
        auto& buffer = params.buffers[i];
        auto root = params.root;
        auto nccl_data_type = getNcclDataType(buffer->type());
        NCCLCHECK(ncclBcast(buffer->data(), buffer->size(), nccl_data_type, root,
                            nccl_param_.nccl_comm_, stream_));
    }
}

void CudaDevice::allReduce(const AllReduceParams& params) {
    if (nccl_param_.world_size_ < 2) {
        return;
    }

    RUNTIME_ASSERT_OP_ARG((int32_t)params.op < ncclRedOp_t::ncclNumOps,
                          "Invalid reduce op: %d", params.op);
    NCCLCHECK(ncclGroupStart());
    for (auto i = 0; i < params.buffers.size(); ++i) {
        auto& buffer = params.buffers[i];
        const auto nccl_op = static_cast<ncclRedOp_t>(params.op);
        const auto nccl_data_type = getNcclDataType(buffer->type());
        NCCLCHECK(ncclAllReduce(buffer->data(), buffer->data(), buffer->size(), nccl_data_type,
                                nccl_op, nccl_param_.nccl_comm_, stream_));
    }
    NCCLCHECK(ncclGroupEnd());
}

void CudaDevice::allGather(const AllGatherParams& params) {
    if (nccl_param_.world_size_ < 2) {
        return;
    }

    NCCLCHECK(ncclGroupStart());
    for (auto i = 0; i < params.buffers.size(); ++i) {
        auto& buffer = params.buffers[i];
        const auto nccl_data_type = getNcclDataType(buffer->type());
        const auto data_num = buffer->size() / nccl_param_.world_size_;
        RUNTIME_ASSERT_OP_ARG(data_num * nccl_param_.world_size_ == buffer->size(),
            "Buffer size %d must be divisible by world size %d",
            buffer->size(), nccl_param_.world_size_);
        const auto data_size = data_num * buffer->typeSize();
        NCCLCHECK(ncclAllGather(buffer->data() + nccl_param_.rank_ * data_size, buffer->data(),
                                data_num, nccl_data_type, nccl_param_.nccl_comm_, stream_));
    }
    NCCLCHECK(ncclGroupEnd());
}

} // namespace fastertransformer
