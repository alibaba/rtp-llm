#include "ATen/ops/cat.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils_torch.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include <memory>
#include <unistd.h>

using namespace std;

namespace fastertransformer {

void CudaDevice::copy(const CopyParams& params) {
    params.check();
    cudaStream_t stream = (params.overlapped && init_params_.enable_comm_overlap) ? communication_stream_ : stream_;
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
        cudaMemcpyAsync(dst.data(), src.data(), src.sizeBytes(), copyType, stream);
    }

    if (copyType == cudaMemcpyDeviceToHost) {
        cudaStreamSynchronize(stream);
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
    cudaStream_t stream = (params.overlapped && init_params_.enable_comm_overlap) ? communication_stream_ : stream_;
    RUNTIME_ASSERT_OP_ARG(shape.size() == 2 || shape.size() == 3,
        "You can only transpose a 2D buffer, but got [%s]", input.debugString().c_str());
    if (shape.size() == 2) {
        auto output = allocateBuffer({data_type, {shape[1], shape[0]}});
        if (output->sizeBytes()) {
            DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(data_type, invokeTransposeAxis01,
                                                output->data(), input.data(), shape[0], shape[1], stream
                                                );
        }
        return output;
    } else {
        auto output = allocateBuffer({data_type, {shape[1], shape[0], shape[2]}});
        if (output->sizeBytes()) {
            DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(data_type, invokeTransposeAxis012,
                                                output->data(), input.data(), shape[0], shape[1], shape[2], stream
                                                );
        }
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
    if (output_shape[0] == 0) {
        return output;
    }
    DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(
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

SplitOutput CudaDevice::split(const SplitParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.input.dim() == 2 && params.dim < params.input.dim()
                              && std::accumulate(params.split_sizes.begin(), params.split_sizes.end(), 0)
                                     == params.input.shape()[params.dim],
                          "split params args error, dim [%ld] split_size_sum [%d] input[%s]",
                          params.dim,
                          std::accumulate(params.split_sizes.begin(), params.split_sizes.end(), 0),
                          params.input.debugString().c_str());
    cudaStream_t stream = (params.overlapped && init_params_.enable_comm_overlap) ? communication_stream_ : stream_;
    std::vector<BufferPtr> outputs;
    size_t                 offset = 0;
    if (params.dim == 0) {
        for (auto& size : params.split_sizes) {
            outputs.emplace_back(clone({params.input.view(offset, size)}));
            offset += size;
        }
    } else {
        vector<size_t> new_shape = params.input.shape();
        for (auto& size : params.split_sizes) {
            new_shape[params.dim] = size;
            BufferPtr output      = allocateBuffer({params.input.type(), new_shape});
            DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(params.input.type(),
                                                invokeSliceDim1Copy,
                                                params.input.data(),
                                                params.input.shape()[0],
                                                params.input.shape()[1],
                                                offset,
                                                size,
                                                output->data(),
                                                stream);
            outputs.emplace_back(output);
            offset += size;
        }
    }
    return {outputs};
}

inline ncclDataType_t getNcclDataType(DataType type) {
    switch (type) {
        case DataType::TYPE_BOOL: return ncclInt8;
        case DataType::TYPE_INT8: return ncclInt8;
        case DataType::TYPE_INT32: return ncclInt32;
        case DataType::TYPE_INT64: return ncclInt64;
        case DataType::TYPE_BYTES: return ncclChar;
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

NcclParam CudaDevice::getNcclParam(ParallelMode mode) {
    switch (mode) {
        case ParallelMode::TP:
            return tp_nccl_param_;
        case ParallelMode::DP:
            return dp_nccl_param_;
        case ParallelMode::DP_AND_TP:
            return dp_tp_nccl_param_;
        default:
            FT_CHECK_WITH_INFO(false, "all reduce not support mode [%d]", mode);
            // avoid compile error
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

void CudaDevice::broadcast(const BroadcastParams& params) {
    FT_CHECK_WITH_INFO(params.mode == ParallelMode::TP, "broadcast not support mode [%d]", params.mode);
    if (tp_nccl_param_.world_size_ < 2) {
        return;
    }
    const auto stream = (params.overlapped && init_params_.enable_comm_overlap)
                      ? communication_stream_
                      : stream_;

    NCCLCHECK(ncclGroupStart());
    for (auto i = 0; i < params.buffers.size(); ++i) {
        auto& buffer = params.buffers[i];
        auto root = params.root;
        auto nccl_data_type = getNcclDataType(buffer->type());
        NCCLCHECK(ncclBcast(buffer->data(), buffer->size(), nccl_data_type, root,
                            tp_nccl_param_.nccl_comm_, stream));
    }
    NCCLCHECK(ncclGroupEnd());
}

AllReduceOutput CudaDevice::allReduce(const AllReduceParams& params) {
    const auto nccl_param = getNcclParam(params.mode);
    if (nccl_param.world_size_ < 2) {
        return AllReduceOutput{params.buffer};
    }

    const auto stream = ((params.overlapped || params.mode == ParallelMode::DP_AND_TP)
                         && init_params_.enable_comm_overlap)
                      ? communication_stream_
                      : stream_;
    if (stream == communication_stream_) {
        // NOTE: before starting communication, we need to make sure that the previous computation
        // has been finished. Otherwise, the communication may overlap with the computation.
        // We use cuda event to ensure the computation on main stream has been finished.
        cudaEvent_t event;
        check_cuda_error(cudaEventCreate(&event));
        check_cuda_error(cudaEventRecord(event, stream_));
        check_cuda_error(cudaStreamWaitEvent(communication_stream_, event, 0));
        check_cuda_error(cudaEventDestroy(event));
    }
    auto& buffer = params.buffer;
    const auto nccl_op = static_cast<ncclRedOp_t>(params.op);
    const auto nccl_data_type = getNcclDataType(buffer->type());

    // if custom allreduce fails, fallback to the default ncclAllReduce
    // dp tmp not support custom_allreduce_comm
    if (params.mode == ParallelMode::TP && custom_allreduce_comm_ && nccl_op == ncclSum
        && custom_allreduce_comm_->checkAllReduceAvailable(buffer->size(), buffer->type(), nccl_param.world_size_)) {
        auto custom_ar_res_buf =
            allocateBuffer({buffer->type(), buffer->shape(), AllocationType::DEVICE}, {"custom_ar_buf"});
        custom_allreduce_comm_->allReduce(
            buffer->data(), custom_ar_res_buf->data(), buffer->size(), buffer->type(), stream);
        return AllReduceOutput{custom_ar_res_buf};
    }

    RUNTIME_ASSERT_OP_ARG((int32_t)params.op < ncclRedOp_t::ncclNumOps,
                          "Invalid reduce op: %d", int(params.op));

    NCCLCHECK(ncclAllReduce(buffer->data(), buffer->data(), buffer->size(), nccl_data_type,
                            nccl_op, nccl_param.nccl_comm_, stream));
    return AllReduceOutput{params.buffer};
}

PrepareAllReduceOutput CudaDevice::prepareAllReduce(const PrepareAllReduceParams& params) {
    auto& nccl_param = tp_nccl_param_;
    if (nccl_param.world_size_ < 2) {
        return PrepareAllReduceOutput{params.buffer};
    }

    auto& buffer = params.buffer;
    if (custom_allreduce_comm_ && static_cast<ncclRedOp_t>(params.op) == ncclSum
        && custom_allreduce_comm_->checkAllReduceAvailable(buffer->size(), buffer->type(), nccl_param.world_size_)) {
        void* custom_ar_buf_ptr = custom_allreduce_comm_->peerCommBufferPtr();
        return PrepareAllReduceOutput{
            BufferPtr(new Buffer(MemoryType::MEMORY_GPU, buffer->type(), buffer->shape(), custom_ar_buf_ptr))};
    }
    return PrepareAllReduceOutput{params.buffer};
}

void computeLengthsAndOffsets(const std::vector<size_t>& split_sizes,
                              const Buffer&              buffer,
                              std::vector<size_t>*       lengths,
                              std::vector<size_t>*       offsets) {
    size_t group_size   = lengths->size();
    bool   equal_splits = false;
    size_t dim0_size    = buffer.shape()[0];
    size_t row_size     = (dim0_size ? buffer.size() / dim0_size : 1);
    size_t split_size   = 0;
    size_t offset       = 0;

    if (split_sizes.empty()) {
        equal_splits = true;
        split_size   = dim0_size / group_size;
    }
    for (int i = 0; i < group_size; ++i) {
        size_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
        (*lengths)[i] = length;
        (*offsets)[i] = offset;
        // TODO: see if we should add overflow protection for offset
        offset += length;
    }
}

AllToAllOutput CudaDevice::allToAll(const AllToAllParams& params) {
    FT_CHECK_WITH_INFO(params.mode == ParallelMode::DP_AND_TP,
                       "all to all just support ParallelMode::DP_AND_TP but got [%d]",
                       params.mode);
    auto&      nccl_param = dp_tp_nccl_param_;
    const auto world_size = nccl_param.world_size_;
    assert(params.buffers.size() > 0);
    if (world_size < 2) {
        return {{params.buffers}};
    }
    auto         stream     = (params.overlapped && init_params_.enable_comm_overlap) ? communication_stream_ : stream_;
    const size_t dims       = params.buffers[0]->dim();
    const auto   batch_size = params.buffers[0]->shape()[0];
    vector<BufferPtr> byte_buffers;
    FT_CHECK_WITH_INFO(dims == 2 || dims == 1,
                       "alltoall just support dims 2 or 1 but got [%s] ",
                       params.buffers[0]->debugString().c_str());
    size_t         dim1_size = 0;
    vector<size_t> dim1_split_size;
    for (const auto& buffer : params.buffers) {
        FT_CHECK_WITH_INFO(
            buffer->dim() == dims && buffer->shape()[0] == batch_size,
            "alltoall all input buffer dims must be consist with dims [%d] and batch_size [%d] but got [%s]",
            dims,
            batch_size,
            buffer->debugString().c_str());
        vector<size_t> new_shape = buffer->shape();
        if (new_shape.size() < 2) {
            new_shape.push_back(1);
        }
        assert(new_shape.size() == 2);
        new_shape[1] *= getTypeSize(buffer->type());
        dim1_size += new_shape[1];
        dim1_split_size.push_back(new_shape[1]);
        byte_buffers.emplace_back(
            std::make_shared<Buffer>(MemoryType::MEMORY_GPU, DataType::TYPE_BYTES, new_shape, buffer->data()));
    }
    BufferPtr input_buffer;
    if (byte_buffers.size() < 2) {
        input_buffer = byte_buffers[0];
    } else {
        input_buffer = allocateBuffer({DataType::TYPE_BYTES, {batch_size, dim1_size}});
        if (batch_size > 0) {
            vector<torch::Tensor> input_tensors;
            for (const auto& buffer : byte_buffers) {
                input_tensors.emplace_back(Buffer2torchTensor(buffer, false));
            }
            torch::Tensor packed_tensor = Buffer2torchTensor(input_buffer, false);
            torch::cat_out(packed_tensor, input_tensors, 1);
        }
    }
    if (stream == communication_stream_) {
        // NOTE: before starting communication, we need to make sure that the previous computation
        // has been finished. Otherwise, the communication may overlap with the computation.
        // We use cuda event to ensure the computation on main stream has been finished.
        cudaEvent_t event;
        check_cuda_error(cudaEventCreate(&event));
        check_cuda_error(cudaEventRecord(event, stream_));
        check_cuda_error(cudaStreamWaitEvent(communication_stream_, event, 0));
        check_cuda_error(cudaEventDestroy(event));
    }
    BufferPtr output;
    if (params.input_split_sizes.size() || params.output_split_sizes.size()) {
        FT_CHECK_WITH_INFO(
            params.input_split_sizes.empty()
                || (params.input_split_sizes.size() == world_size
                    && std::accumulate(params.input_split_sizes.begin(), params.input_split_sizes.end(), 0)
                           == batch_size),
            "alltoall input_split_sizes is not valid");

        if (params.output_split_sizes.empty()) {
            output = allocateBufferLike(*input_buffer);
        } else {
            FT_CHECK_WITH_INFO(params.output_split_sizes.size() == world_size,
                               "alltoall output_split_sizes is not valid");
            size_t output_batch_size =
                std::accumulate(params.output_split_sizes.begin(), params.output_split_sizes.end(), (size_t)0);
            auto new_shape = input_buffer->shape();
            new_shape[0]   = output_batch_size;
            output         = allocateBuffer({input_buffer->type(), new_shape});
        }
        std::vector<size_t> send_lengths(world_size);
        std::vector<size_t> recv_lengths(world_size);
        std::vector<size_t> send_offsets(world_size);
        std::vector<size_t> recv_offsets(world_size);
        computeLengthsAndOffsets(params.input_split_sizes, *input_buffer, &send_lengths, &send_offsets);
        computeLengthsAndOffsets(params.output_split_sizes, *output, &recv_lengths, &recv_offsets);
        all2all_single_unequal_split(input_buffer->data(),
                                     send_lengths.data(),
                                     send_offsets.data(),
                                     output->data(),
                                     recv_lengths.data(),
                                     recv_offsets.data(),
                                     getTypeSize(output->type()),
                                     getNcclDataType(output->type()),
                                     nccl_param.nccl_comm_,
                                     stream);
    } else {
        FT_CHECK_WITH_INFO(input_buffer->shape()[0] % world_size == 0,
                           "all2all_single_equal_split batch size [%d] must divide world size [%d]",
                           input_buffer->shape()[0],
                           world_size);
        output = allocateBufferLike(*input_buffer);
        all2all_single_equal_split(
            input_buffer->data(), output->data(), output->sizeBytes(), nccl_param.nccl_comm_, stream);
    }
    if (byte_buffers.size() < 2) {
        vector<size_t> new_shape = output->shape();
        new_shape[1] /= getTypeSize(params.buffers[0]->type());
        output->updateTypeAndShape(params.buffers[0]->type(), new_shape);
        return {{output}};
    } else {
        vector<BufferPtr> outputs;
        size_t            output_batch_size = output->shape()[0];
        if (output_batch_size == 0) {
            for (int i = 0; i < dim1_split_size.size(); ++i) {
                vector<size_t> new_shape = params.buffers[i]->shape();
                new_shape[0]             = 0;
                outputs.emplace_back(
                    std::make_shared<Buffer>(MemoryType::MEMORY_GPU, params.buffers[i]->type(), new_shape, nullptr));
            }
        } else {
            outputs = split({*output, dim1_split_size, 1, params.overlapped}).outputs;
            for (int i = 0; i < dim1_split_size.size(); ++i) {
                vector<size_t> new_shape = outputs[i]->shape();
                assert(new_shape[0] == output_batch_size);
                new_shape[1] /= getTypeSize(params.buffers[i]->type());
                assert(new_shape[1] == params.buffers[i]->shape()[1]);
                outputs[i]->updateTypeAndShape(params.buffers[i]->type(), new_shape);
            }
        }
        return {outputs};
    }
}

void CudaDevice::allGather(const AllGatherParams& params) {
    cudaStream_t stream = (params.overlapped && init_params_.enable_comm_overlap) ? communication_stream_ : stream_;
    if (params.mode == ParallelMode::DP_AND_TP && dp_nccl_param_.world_size_ != dp_tp_nccl_param_.world_size_) {
        // 两种实现，一种是先dp gather 然后再tp broast
        // 还有一种是直接all gather, 然后再transpose，再取第0个
        // 目前是走第一种
        allGather({params.buffers, ParallelMode::DP, params.overlapped});
        broadcast({params.buffers, 0, ParallelMode::TP, params.overlapped});
    } else {
        auto nccl_param = getNcclParam(params.mode);
        if (nccl_param.world_size_ < 2) {
            return;
        }
        NCCLCHECK(ncclGroupStart());
        for (auto i = 0; i < params.buffers.size(); ++i) {
            auto& buffer = params.buffers[i];
            const auto nccl_data_type = getNcclDataType(buffer->type());
            const auto data_num = buffer->size() / nccl_param.world_size_;
            RUNTIME_ASSERT_OP_ARG(data_num * nccl_param.world_size_ == buffer->size(),
                                  "Buffer size %ld must be divisible by world size %d",
                                  buffer->size(), nccl_param.world_size_);
            const auto data_size = data_num * buffer->typeSize();
            NCCLCHECK(ncclAllGather((char*)(buffer->data()) + nccl_param.rank_ * data_size, buffer->data(),
                                    data_num, nccl_data_type, nccl_param.nccl_comm_, stream));
        }
        NCCLCHECK(ncclGroupEnd());
    }
}

} // namespace fastertransformer
