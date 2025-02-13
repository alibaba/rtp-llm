#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils_torch.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"


using namespace std;

namespace fastertransformer {

void CudaDevice::copy(const CopyParams& params) {
    params.check();
    if (params.dst.isQBuffer() && params.src.isQBuffer()) {
        auto dst_ptr = reinterpret_cast<const QBuffer*>(&params.dst);
        auto src_ptr = reinterpret_cast<const QBuffer*>(&params.src);
        copy({dst_ptr->kernel(), src_ptr->kernel()});
        copy({dst_ptr->scales(), src_ptr->scales()});
        copy({dst_ptr->zeros(), src_ptr->zeros()});
        return;
    }

    const auto& src = params.src;
    const auto& dst = params.dst;

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

    if (copyType == cudaMemcpyHostToHost) {
        std::memcpy(dst.data(), src.data(), src.sizeBytes());
    } else {
        cudaMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), copyType, stream_);
    }

    if (copyType == cudaMemcpyDeviceToHost) {
        cudaStreamSynchronize(stream_);
        check_cuda_error(cudaGetLastError());
    }

    sync_check_cuda_error();
}

void CudaDevice::noBlockCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
    cudaMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), cudaMemcpyDefault, no_block_copy_stream_);
    cudaStreamSynchronize(no_block_copy_stream_);
    sync_check_cuda_error();
}

TransposeOutput CudaDevice::transpose(const TransposeParams& params) {
    const auto& input = params.input;
    const auto data_type = input.type();
    const auto shape = input.shape();

    RUNTIME_ASSERT_OP_ARG(shape.size() == 2 || shape.size() == 3,
        "You can only transpose a 2D buffer, but got [%s]", input.debugString().c_str());
    if (shape.size() == 2) {
        auto output = allocateBuffer({data_type, {shape[1], shape[0]}});
        DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(data_type, invokeTransposeAxis01,
                                            output->data(), input.data(), shape[0], shape[1], stream_
                                            );
        return output;
    } else {
        auto output = allocateBuffer({data_type, {shape[1], shape[0], shape[2]}});
        DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(data_type, invokeTransposeAxis012,
                                            output->data(), input.data(), shape[0], shape[1], shape[2], stream_
                                            );
        return output;
    }
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
SPECIALIZE_CAST_TO(__half, int8_t);
SPECIALIZE_CAST_TO(__half, uint8_t);
SPECIALIZE_CAST_TO(__half, int32_t);
SPECIALIZE_CAST_TO(__half, uint32_t);
SPECIALIZE_CAST_TO(__half, int64_t);
SPECIALIZE_CAST_TO(__half, uint64_t);
SPECIALIZE_CAST_TO(__nv_bfloat16, int64_t);
SPECIALIZE_CAST_TO(__nv_bfloat16, uint64_t);
SPECIALIZE_CAST_TO(__nv_bfloat16, __half);
SPECIALIZE_CAST_TO(__half, __nv_bfloat16);
SPECIALIZE_CAST_TO(__nv_bfloat16, int8_t)
SPECIALIZE_CAST_TO(__nv_bfloat16, uint8_t)
SPECIALIZE_CAST_TO(__nv_bfloat16, int32_t)
SPECIALIZE_CAST_TO(__nv_bfloat16, uint32_t)

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
    return {output};
}

SelectOutput CudaDevice::select(const SelectParams& params) {
    if ((params.input.where() != MemoryType::MEMORY_GPU) || (params.dim > 0)) {
        return DeviceBase::select(params);
    }

    RUNTIME_ASSERT_OP_ARG(params.index.type() == DataType::TYPE_INT32, "Select index must be int32.");
    RUNTIME_ASSERT_OP_ARG(params.dim == 0, "select op tmp only support dim == 0");

    const auto& input = params.input;
    auto output_shape = input.shape();
    output_shape[0] = params.index.size();
    auto num_selected_element = input.size() / input.shape()[0];
    auto output = allocateBuffer({input.type(), output_shape});
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        input.type(),
        invokeLookupHiddenStateOfLastToken,
        output->data(),
        input.data(),
        (int*)params.index.data(),
        (int)params.index.size(),
        num_selected_element,
        0,
        stream_);
    return output;
}

MultiplyOutput CudaDevice::multiply(const MultiplyParams& params) {
    const auto& A = params.A;
    const auto& B = params.B;

    int m, n;
    if (A.shape() == B.shape()) {
        m = A.size();
        n = 1;
    } else if (A.size() == 1) {
        m = 1;
        n = B.size();
    } else if (A.shape().size() == 1 && B.shape()[0] == A.shape()[0]) {
        m = A.shape()[0];
        n = B.size() / m;
    } else {
        RUNTIME_ASSERT_OP_ARG(false,
            "multiply can not be applied to A[%s] and B[%s]",
            A.debugString().c_str(), B.debugString().c_str());
    }

    RUNTIME_ASSERT_OP_ARG(A.type() == B.type(),
                          "A and B must have same type, but got %d vs %d", A.type(), B.type());
    const auto data_type = A.type();

    BufferPtr output;
    if (params.output) {
        output = params.output;
        RUNTIME_ASSERT_OP_ARG(output->type() == data_type,
                              "Output type must be same as A and B, but got %d vs %d",
                              output->type(), data_type);
        RUNTIME_ASSERT_OP_ARG(output->shape()[0] == n,
                              "Output 0-d size must be %d, but got %ld", n, output->shape()[0]);
        RUNTIME_ASSERT_OP_ARG(output->size() == B.size(),
                              "Output size must be %ld, but got %ld", B.size(), output->size());
    } else {
        output = allocateBuffer({data_type, B.shape()});
    }

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        data_type,
        invokeScaledDot,
        output->data(),
        B.data(),
        A.data(),
        m,
        n,
        stream_
    );

    printBufferData(*output, "multiply_output");

    return output;
}

inline ncclDataType_t getNcclDataType(DataType type) {
    switch (type) {
        case DataType::TYPE_BOOL: return ncclInt8;
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
    NCCLCHECK(ncclGroupStart());
    for (auto i = 0; i < params.buffers.size(); ++i) {
        auto& buffer = params.buffers[i];
        auto root = params.root;
        auto nccl_data_type = getNcclDataType(buffer->type());
        NCCLCHECK(ncclBcast(buffer->data(), buffer->size(), nccl_data_type, root,
                            nccl_param_.nccl_comm_, stream_));
    }
    NCCLCHECK(ncclGroupEnd());
}

AllReduceOutput CudaDevice::allReduce(const AllReduceParams& params) {
    if (nccl_param_.world_size_ < 2) {
        return AllReduceOutput{params.buffer};
    }
    auto& buffer = params.buffer;
    const auto nccl_op = static_cast<ncclRedOp_t>(params.op);
    const auto nccl_data_type = getNcclDataType(buffer->type());

    // if custom allreduce fails, fallback to the default ncclAllReduce
    if (custom_allreduce_comm_ && nccl_op == ncclSum
        && custom_allreduce_comm_->checkAllReduceAvailable(buffer->size(), buffer->type(), nccl_param_.world_size_)) {
        auto custom_ar_res_buf =
            allocateBuffer({buffer->type(), buffer->shape(), AllocationType::DEVICE}, {"custom_ar_buf"});
        custom_allreduce_comm_->allReduce(
            buffer->data(), custom_ar_res_buf->data(), buffer->size(), buffer->type(), stream_);
        return AllReduceOutput{custom_ar_res_buf};
    }

    RUNTIME_ASSERT_OP_ARG((int32_t)params.op < ncclRedOp_t::ncclNumOps,
                          "Invalid reduce op: %d", int(params.op));

    NCCLCHECK(ncclAllReduce(buffer->data(), buffer->data(), buffer->size(), nccl_data_type,
                            nccl_op, nccl_param_.nccl_comm_, stream_));
    return AllReduceOutput{params.buffer};
}

PrepareAllReduceOutput CudaDevice::prepareAllReduce(const PrepareAllReduceParams& params) {
    if (nccl_param_.world_size_ < 2) {
        return PrepareAllReduceOutput{params.buffer};
    }

    auto& buffer = params.buffer;
    if (custom_allreduce_comm_ && static_cast<ncclRedOp_t>(params.op) == ncclSum &&
        custom_allreduce_comm_->checkAllReduceAvailable(buffer->size(), buffer->type(), nccl_param_.world_size_)) {
        void* custom_ar_buf_ptr = custom_allreduce_comm_->peerCommBufferPtr();
        return PrepareAllReduceOutput{
            BufferPtr(new Buffer(MemoryType::MEMORY_GPU,
                buffer->type(),
                buffer->shape(),
                custom_ar_buf_ptr))
        };
    }
    return PrepareAllReduceOutput{params.buffer};
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
            "Buffer size %ld must be divisible by world size %d",
            buffer->size(), nccl_param_.world_size_);
        const auto data_size = data_num * buffer->typeSize();
        NCCLCHECK(ncclAllGather((char*)(buffer->data()) + nccl_param_.rank_ * data_size, buffer->data(),
                                data_num, nccl_data_type, nccl_param_.nccl_comm_, stream_));
    }
    NCCLCHECK(ncclGroupEnd());
}

} // namespace fastertransformer
