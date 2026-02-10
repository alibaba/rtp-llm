#include "custom_ar_comm.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "c10/hip/HIPStream.h"
#include "c10/hip/HIPGraphsC10Utils.h"
#include <climits>
#include <cstdint>
#include <sys/types.h>
#include <iostream>
#include <unordered_set>
#include <vector>
using namespace std;

namespace rtp_llm {

CustomAllReduceComm::CustomAllReduceComm(const std::vector<size_t>& tp_ranks, size_t rank, size_t rank_index, const HWKernelConfig& hw_kernel_config):
    rank_(rank),
    rank_index_(rank_index),
    world_size_(tp_ranks.size()),
    support_nv_link_(true),  // TODO(liyangcheng.lyc): add check function
    comm_buf_threshold_(getCommBufThreshold()),
    tp_ranks_(std::move(tp_ranks)),
    ft_disable_custom_ar_(hw_kernel_config.ft_disable_custom_ar),
    rocm_disable_custom_ag_(hw_kernel_config.rocm_disable_custom_ag) {}

CustomAllReduceComm::~CustomAllReduceComm() {
    aiter::dispose(fa_);
    fa_ = 0;
}

bool CustomAllReduceComm::checkAllReduceAvailable(size_t elts_total_num, DataType data_type, size_t world_size) {
    size_t elts_total_size = elts_total_num * getTypeSize(data_type);
    if (elts_total_size % 16 != 0) {
        return false;
    }

    if (world_size == 2 or support_nv_link_) {
        return elts_total_size <= comm_buf_threshold_;
    }

    return false;
}

bool CustomAllReduceComm::checkAllGatherAvailable() {
    if (rocm_disable_custom_ag_) {
        RTP_LLM_LOG_INFO("Disable custom ag since ROCM_DISABLE_CUSTOM_AG is set");
        return false;
    }

    return true;
}

void CustomAllReduceComm::allReduce(torch::Tensor& input_tensor, torch::Tensor& output_tensor) {
    if (at::hip::currentStreamCaptureStatusMayInitCtx() != at::hip::CaptureStatus::None) {
        aiter::all_reduce(fa_, input_tensor, output_tensor, false, false, std::nullopt);
    } else {
         aiter::all_reduce(fa_, input_tensor, output_tensor, false, false, buffer_);
    }
}

void CustomAllReduceComm::allGather(torch::Tensor& input_tensor, torch::Tensor& output_tensor) {
    if (at::hip::currentStreamCaptureStatusMayInitCtx() != at::hip::CaptureStatus::None) {
        aiter::all_gather_reg(fa_, input_tensor, output_tensor);
    } else {
        aiter::all_gather_unreg(fa_, input_tensor, buffer_, output_tensor);
    }
}

void CustomAllReduceComm::registerGraphBuffers() {
    auto handle_and_offset = aiter::get_graph_buffer_ipc_meta(fa_); // tuple<tensor, vector<int64_t>> -> vector<tensor> size=2
    auto handle = std::get<0>(handle_and_offset);
    auto offset = std::get<1>(handle_and_offset);

    auto _handles = all_gather(handle.data_ptr(), handle.element_size() * handle.numel(), at::hip::getCurrentHIPStream().stream());
    auto _offsets = all_gather(offset.data_ptr(), offset.element_size() * offset.numel(), at::hip::getCurrentHIPStream().stream());
    std::vector<torch::Tensor> handles(world_size_); // vector<string>          -> vector<tensor>
    std::vector<torch::Tensor> offsets(world_size_); // vector<vector<int64_t>> -> vector<tensor>
    for (int i = 0; i < world_size_; ++i) {
        handles[i] = torch::from_blob(_handles[i].data(), handle.sizes(), handle.dtype());
        offsets[i] = torch::from_blob(_offsets[i].data(), offset.sizes(), offset.dtype());
    }
    aiter::register_graph_buffers(fa_, handles, offsets);
}

std::vector<std::vector<char>> CustomAllReduceComm::all_gather(void* addr, size_t size, hipStream_t stream) {
    char* device_buffer;
    ROCM_CHECK(hipMalloc(&device_buffer, size * world_size_));
    ROCM_CHECK(hipMemcpyAsync(device_buffer + rank_index_ * size, addr, size, hipMemcpyHostToDevice, stream));
    ftNcclAllGather(device_buffer, device_buffer, size, rank_index_, nccl_para_, stream);
    ROCM_CHECK(hipStreamSynchronize(stream));
    std::vector<std::vector<char>> ret(world_size_);
    for (size_t i = 0; i < world_size_; ++i) {
        std::vector<char> tmp(size);
        ROCM_CHECK(hipMemcpyAsync(tmp.data(), device_buffer + size * i, size, hipMemcpyDeviceToHost, stream));
        ret[i] = tmp;
    }
    ROCM_CHECK(hipFreeAsync(device_buffer, stream));
    return ret;
}

void CustomAllReduceComm::init(const NcclParam& nccl_para, hipStream_t stream) {
    // prepare share buffer

    // meta data buffers need to be "uncached" for signal on MI200
    meta_   = aiter::allocate_meta_buffer(aiter::meta_size() + comm_buf_threshold_);
    void* raw_ptr;
    ROCM_CHECK(hipMalloc(&raw_ptr, comm_buf_threshold_));
    auto deleter = [](void* p) { hipFree(p); };
    buffer_ = torch::from_blob(raw_ptr, {comm_buf_threshold_}, deleter, torch::kCUDA);
    rank_data_ = torch::empty({16 * 1024 * 1024}, torch::dtype(torch::kUInt8).device(torch::kCUDA));

    std::vector<torch::Tensor> meta_handles   = prepareP2PBuffer_(nccl_para, meta_, stream);
    std::vector<torch::Tensor> buffer_handles = prepareP2PBuffer_(nccl_para, buffer_, stream);

    std::vector<int64_t> meta_offsets(world_size_, 0);
    std::vector<int64_t> buffer_offsets(world_size_, 0);

    fa_ = aiter::init_custom_ar(meta_, rank_data_, meta_handles, meta_offsets, rank_index_, support_nv_link_);

    aiter::register_buffer(fa_, buffer_, buffer_handles, buffer_offsets);
    nccl_para_ = nccl_para;
}

std::vector<torch::Tensor>
CustomAllReduceComm::prepareP2PBuffer_(const NcclParam& nccl_para, torch::Tensor& local_buffer, hipStream_t stream) {
    // malloc serial handle buffer
    char* serial_handle_buffer_ptr;
    ROCM_CHECK(hipMalloc(&serial_handle_buffer_ptr, IPChandleBufSize(world_size_)));

    // open local hipIpcMemHandle
    torch::Tensor     local_buffer_handle_tensor = aiter::get_meta_buffer_ipc_handle(local_buffer);
    hipIpcMemHandle_t local_buffer_handle        = *(hipIpcMemHandle_t*)local_buffer_handle_tensor.data_ptr();

    // serialized hipIpcMemHandle
    ROCM_CHECK(hipMemcpyAsync(serial_handle_buffer_ptr + HIP_IPC_HANDLE_SIZE * rank_index_,
                              local_buffer_handle.reserved,
                              HIP_IPC_HANDLE_SIZE,
                              hipMemcpyHostToDevice,
                              stream));

    // all gather serialized hipIpcMemHandle
    ftNcclAllGather(
        serial_handle_buffer_ptr, serial_handle_buffer_ptr, HIP_IPC_HANDLE_SIZE, rank_index_, nccl_para, stream);
    ROCM_CHECK(hipStreamSynchronize(stream));

    // deserialize all ranks' hipIpcMemHandle, and convert to std::tensor for aiter use
    std::vector<torch::Tensor> handles(world_size_);
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    for (size_t i = 0; i < handles.size(); ++i) {
        char tmp[HIP_IPC_HANDLE_SIZE];
        handles[i] = torch::empty({static_cast<int64_t>(HIP_IPC_HANDLE_SIZE)}, options);
        ROCM_CHECK(hipMemcpyAsync(handles[i].data_ptr(),
                                   serial_handle_buffer_ptr + HIP_IPC_HANDLE_SIZE * i,
                                   HIP_IPC_HANDLE_SIZE,
                                   hipMemcpyDeviceToHost,
                                   stream));
    }

    ROCM_CHECK(hipFreeAsync(serial_handle_buffer_ptr, stream));
    return handles;
}

bool CustomAllReduceComm::shouldCustomAR(const std::vector<size_t>& tp_ranks, size_t rank, const HWKernelConfig& hw_kernel_config) {
    size_t world_size       = tp_ranks.size();
    size_t local_world_size = rocm::getDeviceCount();

    // 1. check whether all ranks are on same nodes
    if (world_size > local_world_size) {
        RTP_LLM_LOG_INFO(
            "Disable custom ar since TP is performanced on multi nodes, world_size=%d, local_world_size=%d",
            world_size,
            local_world_size);
        return false;
    }

    // 2. check whether disabled flag is set
    if (hw_kernel_config.ft_disable_custom_ar) {
        RTP_LLM_LOG_INFO("Disable custom ar since FT_DISABLE_CUSTOM_AR is set");
        return false;
    }

    // 3. check whether world size is valid
    std::unordered_set<size_t> available_world_sizes = {2, 4, 6, 8};
    if (available_world_sizes.find(world_size) == available_world_sizes.end()) {
        RTP_LLM_LOG_INFO("Disable custom ar for invalid world size %d", world_size);
        return false;
    }

    // 4. TODO(liyangcheng.lyc) check whether nvlink is fully connected

    // 5. check whether p2p capability is good
    // On AMD GPU, p2p is always enabled between XGMI connected GPUs

    return true;
}

size_t CustomAllReduceComm::getCommBufThreshold() {
    int64_t custom_ar_size_threshold = 8192 * 1024 * 16;
    return custom_ar_size_threshold;
}

std::unique_ptr<CustomAllReduceComm>
initCustomAllReduceComm(const NcclParam& nccl_para, const std::vector<size_t>& tp_ranks, hipStream_t stream, const HWKernelConfig& hw_kernel_config) {
    size_t rank_index = 0;
    for (size_t i = 0; i < tp_ranks.size(); i++) {
        if (tp_ranks[i] == nccl_para.rank_) {
            rank_index = i;
            break;
        }
    }

    if (!CustomAllReduceComm::shouldCustomAR(tp_ranks, nccl_para.rank_, hw_kernel_config)) {
        return nullptr;
    }

    auto comm = std::make_unique<CustomAllReduceComm>(tp_ranks, nccl_para.rank_, rank_index, hw_kernel_config);
    comm->init(nccl_para, stream);
    RTP_LLM_LOG_INFO("Custom all reduce is enabled on rank %d of %d", nccl_para.rank_, tp_ranks.size());
    return comm;
}

}  // namespace rtp_llm
