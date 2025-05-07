#include "maga_transformer/cpp/devices/rocm_impl/ROCmDevice.h"
#include "maga_transformer/cpp/devices/CommonDefines.h"
#include "maga_transformer/cpp/devices/utils/DebugUtils.h"
#include "maga_transformer/cpp/cuda/Dispatch.h"
#include "maga_transformer/cpp/utils/compiler_config.h"
#include "maga_transformer/cpp/cuda/nccl/nccl_utils_torch.h"
#include "maga_transformer/cpp/cuda/nccl/nccl_utils.h"

namespace rtp_llm {

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

void ROCmDevice::broadcast(const BroadcastParams& params) {
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

AllReduceOutput ROCmDevice::allReduce(const AllReduceParams& params) {
    if (nccl_param_.world_size_ < 2) {
        return {params.buffer};
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
                          "Invalid reduce op: %d", params.op);
 
    NCCLCHECK(ncclAllReduce(buffer->data(), buffer->data(), buffer->size(), nccl_data_type,
                            nccl_op, nccl_param_.nccl_comm_, stream_));
    return {params.buffer};
}

PrepareAllReduceOutput ROCmDevice::prepareAllReduce(const PrepareAllReduceParams& params) {
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

void ROCmDevice::allGather(const AllGatherParams& params) {
    if (nccl_param_.world_size_ < 2) {
        return;
    }

    NCCLCHECK(ncclGroupStart());
    for (auto i = 0; i < params.recv_buffers.size(); ++i) {
        auto& buffer = params.recv_buffers[i];
        const auto nccl_data_type = getNcclDataType(buffer->type());
        const auto data_num = buffer->size() / nccl_param_.world_size_;
        RUNTIME_ASSERT_OP_ARG(data_num * nccl_param_.world_size_ == buffer->size(),
            "Buffer size %d must be divisible by world size %d",
            buffer->size(), nccl_param_.world_size_);
        const auto data_size = data_num * buffer->typeSize();
        NCCLCHECK(ncclAllGather(static_cast<char*>(buffer->data()) + nccl_param_.rank_ * data_size, buffer->data(),
                                data_num, nccl_data_type, nccl_param_.nccl_comm_, stream_));
    }
    NCCLCHECK(ncclGroupEnd());
}

} // namespace rtp_llm
