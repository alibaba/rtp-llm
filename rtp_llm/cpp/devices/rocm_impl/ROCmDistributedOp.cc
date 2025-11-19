#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils_torch.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/kernels/moe_kernels.h"

using namespace std;

namespace rtp_llm {

inline ncclDataType_t getNcclDataType(DataType type) {
    switch (type) {
        case DataType::TYPE_BOOL:
            return ncclInt8;
        case DataType::TYPE_INT8:
            return ncclInt8;
        case DataType::TYPE_INT32:
            return ncclInt32;
        case DataType::TYPE_INT64:
            return ncclInt64;
        case DataType::TYPE_BYTES:
            return ncclChar;
        case DataType::TYPE_UINT8:
            return ncclUint8;
        case DataType::TYPE_UINT32:
            return ncclUint32;
        case DataType::TYPE_UINT64:
            return ncclUint64;
        case DataType::TYPE_FP16:
            return ncclFloat16;
        case DataType::TYPE_FP32:
            return ncclFloat32;
        case DataType::TYPE_FP64:
            return ncclFloat64;
        case DataType::TYPE_BF16:
            return ncclBfloat16;
        default:
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

NcclParam ROCmDevice::getNcclParam(ParallelMode mode) {
    switch (mode) {
        case ParallelMode::TP:
            return tp_nccl_param_;
        case ParallelMode::DP:
            return dp_nccl_param_;
        case ParallelMode::DP_AND_TP:
            return dp_tp_nccl_param_;
        case ParallelMode::FFN_TP:
            return ffn_tp_nccl_param_;
        case ParallelMode::EP:
            return dp_tp_nccl_param_;
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "all reduce not support mode [%d]", mode);
            // avoid compile error
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

void ROCmDevice::broadcast(const BroadcastParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.mode == ParallelMode::TP || params.mode == ParallelMode::DP_AND_TP,
        "broadcast not support mode [%d]", params.mode);
    if (tp_nccl_param_.world_size_ < 2) {
        return;
    }
    const auto stream = (params.overlapped && init_params_.enable_comm_overlap) ? communication_stream_ : stream_;

    NCCLCHECK(ncclGroupStart());
    for (auto i = 0; i < params.buffers.size(); ++i) {
        auto& buffer         = params.buffers[i];
        auto  root           = params.root;
        auto  nccl_data_type = getNcclDataType(buffer->type());
        NCCLCHECK(ncclBcast(buffer->data(), buffer->size(), nccl_data_type, root, tp_nccl_param_.nccl_comm_, stream));
    }
    NCCLCHECK(ncclGroupEnd());
}

AllReduceOutput ROCmDevice::allReduce(const AllReduceParams& params) {
    NcclParam nccl_param = getNcclParam(params.mode);
    if (nccl_param.world_size_ < 2) {
        return AllReduceOutput{params.buffer};
    }

    const auto stream =
        ((params.overlapped || params.mode == ParallelMode::DP_AND_TP) && init_params_.enable_comm_overlap) ?
            communication_stream_ :
            stream_;
    if (stream == communication_stream_) {
        // NOTE: before starting communication, we need to make sure that the previous computation
        // has been finished. Otherwise, the communication may overlap with the computation.
        // We use cuda event to ensure the computation on main stream has been finished.
        overlappedComputeBarrier();
    }
    auto&      buffer         = params.buffer;
    const auto nccl_op        = static_cast<ncclRedOp_t>(params.op);
    const auto nccl_data_type = getNcclDataType(buffer->type());

    bool use_quick_ar =
        !params.dest
        && (params.mode == ParallelMode::TP
            || (params.mode == ParallelMode::FFN_TP && tp_nccl_param_ == ffn_tp_nccl_param_))
        && quick_allreduce_comm_ && nccl_op == ncclSum
        && quick_allreduce_comm_->checkAllReduceAvailable(buffer->size(), buffer->type(), nccl_param.world_size_);

    // if quick allreduce fails, try custom allreduce then
    if (use_quick_ar) {
        auto quick_ar_res_buf =
            allocateBuffer({buffer->type(), buffer->shape(), AllocationType::DEVICE}, {"quick_ar_buf"});
        torch::Tensor input_tensor  = Buffer2torchTensor(*buffer, false);
        torch::Tensor output_tensor = Buffer2torchTensor(*quick_ar_res_buf, false);
        quick_allreduce_comm_->allReduce(input_tensor, output_tensor);
        return AllReduceOutput{quick_ar_res_buf};
    }

    bool use_custom_ar =
        !params.dest
        && (params.mode == ParallelMode::TP
            || (params.mode == ParallelMode::FFN_TP && tp_nccl_param_ == ffn_tp_nccl_param_))
        && custom_allreduce_comm_ && nccl_op == ncclSum
        && custom_allreduce_comm_->checkAllReduceAvailable(buffer->size(), buffer->type(), nccl_param.world_size_);

    // if custom allreduce fails, fallback to the default ncclAllReduce
    // dp tmp not support custom_allreduce_comm
    if (use_custom_ar) {
        auto custom_ar_res_buf =
            allocateBuffer({buffer->type(), buffer->shape(), AllocationType::DEVICE}, {"custom_ar_buf"});
        torch::Tensor input_tensor  = Buffer2torchTensor(*buffer, false);
        torch::Tensor output_tensor = Buffer2torchTensor(*custom_ar_res_buf, false);
        custom_allreduce_comm_->allReduce(input_tensor, output_tensor);
        return AllReduceOutput{custom_ar_res_buf};
    }

    RUNTIME_ASSERT_OP_ARG((int32_t)params.op < ncclRedOp_t::ncclNumOps, "Invalid reduce op: %d", int(params.op));

    auto& dest_buffer = params.dest ? params.dest : buffer;
    NCCLCHECK(ncclAllReduce(
        buffer->data(), dest_buffer->data(), buffer->size(), nccl_data_type, nccl_op, nccl_param.nccl_comm_, stream));
    return AllReduceOutput{params.buffer};
}

PrepareAllReduceOutput ROCmDevice::prepareAllReduce(const PrepareAllReduceParams& params) {
    // (liyangcheng.lyc): ROCm custom allreduce does not need to prepare another comm buffer
    return PrepareAllReduceOutput{params.buffer};
}

void ROCmDevice::allGather(const AllGatherParams& params) {
    hipStream_t stream     = (params.overlapped && init_params_.enable_comm_overlap) ? communication_stream_ : stream_;
    NcclParam   nccl_param = getNcclParam(params.mode);
    if (nccl_param.world_size_ < 2) {
        return;
    }
    NCCLCHECK(ncclGroupStart());
    for (auto i = 0; i < params.recv_buffers.size(); ++i) {
        auto&      recv_buffer    = params.recv_buffers[i];
        const auto nccl_data_type = getNcclDataType(recv_buffer->type());
        const auto data_num       = recv_buffer->size() / nccl_param.world_size_;
        RUNTIME_ASSERT_OP_ARG(data_num * nccl_param.world_size_ == recv_buffer->size(),
                              "Buffer size %ld must be divisible by world size %d",
                              recv_buffer->size(),
                              nccl_param.world_size_);
        if (params.inplace) {
            const auto data_size = data_num * recv_buffer->typeSize();
            NCCLCHECK(ncclAllGather((char*)(recv_buffer->data()) + nccl_param.rank_ * data_size,
                                    recv_buffer->data(),
                                    data_num,
                                    nccl_data_type,
                                    nccl_param.nccl_comm_,
                                    stream));
        } else {
            auto& send_buffer = params.send_buffers[i];
            NCCLCHECK(ncclAllGather(
                send_buffer->data(), recv_buffer->data(), data_num, nccl_data_type, nccl_param.nccl_comm_, stream));
        }
    }
    NCCLCHECK(ncclGroupEnd());
}

// TODO: allToAll, allGather, prepareAllReduce, allReduce
SplitOutput ROCmDevice::split(const SplitParams& params) {
    RUNTIME_ASSERT_OP_ARG(params.input.dim() == 2 && params.dim < params.input.dim()
                              && std::accumulate(params.split_sizes.begin(), params.split_sizes.end(), 0)
                                     == params.input.shape()[params.dim],
                          "split params args error, dim [%ld] split_size_sum [%d] input[%s]",
                          params.dim,
                          std::accumulate(params.split_sizes.begin(), params.split_sizes.end(), 0),
                          params.input.debugString().c_str());
    hipStream_t stream = (params.overlapped && init_params_.enable_comm_overlap) ? communication_stream_ : stream_;
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

AllToAllOutput ROCmDevice::allToAll(const AllToAllParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.mode == ParallelMode::DP_AND_TP,
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
    RTP_LLM_CHECK_WITH_INFO(dims == 2 || dims == 1,
                            "alltoall just support dims 2 or 1 but got [%s] ",
                            params.buffers[0]->debugString().c_str());
    size_t         dim1_size = 0;
    vector<size_t> dim1_split_size;
    for (const auto& buffer : params.buffers) {
        RTP_LLM_CHECK_WITH_INFO(
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
        hipEvent_t event;
        ROCM_CHECK(hipEventCreate(&event));
        ROCM_CHECK(hipEventRecord(event, stream_));
        ROCM_CHECK(hipStreamWaitEvent(communication_stream_, event, 0));
        ROCM_CHECK(hipEventDestroy(event));
    }
    BufferPtr output;
    if (params.input_split_sizes.size() || params.output_split_sizes.size()) {
        RTP_LLM_CHECK_WITH_INFO(
            params.input_split_sizes.empty()
                || (params.input_split_sizes.size() == world_size
                    && std::accumulate(params.input_split_sizes.begin(), params.input_split_sizes.end(), 0)
                           == batch_size),
            "alltoall input_split_sizes is not valid");

        if (params.output_split_sizes.empty()) {
            output = allocateBufferLike(*input_buffer);
        } else {
            RTP_LLM_CHECK_WITH_INFO(params.output_split_sizes.size() == world_size,
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
        RTP_LLM_CHECK_WITH_INFO(input_buffer->shape()[0] % world_size == 0,
                                "all2all_single_equal_split batch size [%d] must divide world size [%d]",
                                input_buffer->shape()[0],
                                world_size);
        output = allocateBufferLike(*input_buffer);
        all2all_single_equal_split(
            input_buffer->data(), output->data(), output->sizeBytes(), nccl_param.nccl_comm_, stream);
    }
    AllToAllOutput all_to_all_output;
    if (byte_buffers.size() < 2) {
        vector<size_t> new_shape = output->shape();
        new_shape[1] /= getTypeSize(params.buffers[0]->type());
        output->updateTypeAndShape(params.buffers[0]->type(), new_shape);
        all_to_all_output = {{output}};
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
        all_to_all_output = {{outputs}, input_buffer, output};
    }
    if (params.overlapped) {
        all_to_all_output.comm_barrier_hook = createCommHook();
    }
    return all_to_all_output;
}

}  // namespace rtp_llm
