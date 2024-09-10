#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils_torch.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"

namespace fastertransformer {

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

    RUNTIME_ASSERT_OP_ARG((int32_t)params.op < ncclRedOp_t::ncclNumOps,
                          "Invalid reduce op: %d", params.op);
 
    NCCLCHECK(ncclAllReduce(buffer->data(), buffer->data(), buffer->size(), nccl_data_type,
                            nccl_op, nccl_param_.nccl_comm_, stream_));
    return {params.buffer};
}

void ROCmDevice::allGather(const AllGatherParams& params) {
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
        NCCLCHECK(ncclAllGather(static_cast<char*>(buffer->data()) + nccl_param_.rank_ * data_size, buffer->data(),
                                data_num, nccl_data_type, nccl_param_.nccl_comm_, stream_));
    }
    NCCLCHECK(ncclGroupEnd());
}

} // namespace fastertransformer
