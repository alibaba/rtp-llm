#pragma once

#include <cstddef>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/all.h>

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"

// aiter custom all reduce kernel
#include "custom_all_reduce.h"
// #include "aiter_meta/csrc/include/custom_all_reduce.h"

namespace rtp_llm {
class CustomAllReduceComm {
public:
    CustomAllReduceComm(const std::vector<size_t>& tp_ranks, size_t rank, size_t rank_index);

    ~CustomAllReduceComm();

    void init(const NcclParam& nccl_para, hipStream_t stream);

    void allReduce(torch::Tensor& input_tensor, torch::Tensor& output_tensor);

    bool checkAllReduceAvailable(size_t elts_total_num, DataType data_type, size_t world_size);

    static bool shouldCustomAR(const std::vector<size_t>& tp_ranks, size_t rank);

    void registerGraphBuffers();

    std::vector<std::vector<char>> all_gather(void* addr, size_t size, hipStream_t stream);

private:
    static size_t getCommBufThreshold();

    size_t IPChandleBufSize(size_t world_size) const {
        return HIP_IPC_HANDLE_SIZE * world_size;
    }

    std::vector<torch::Tensor>
    prepareP2PBuffer_(const NcclParam& nccl_para, torch::Tensor& local_buffer, hipStream_t stream);

    const size_t        rank_               = 0;
    const size_t        rank_index_         = 0;
    const size_t        world_size_         = 0;
    bool                support_nv_link_    = false;
    const int64_t       comm_buf_threshold_ = 0;
    std::vector<size_t> tp_ranks_;
    torch::Tensor       meta_;
    torch::Tensor       buffer_;
    torch::Tensor       rank_data_;
    int64_t             fa_;
    NcclParam           nccl_para_;
};

std::unique_ptr<CustomAllReduceComm>
initCustomAllReduceComm(const NcclParam& nccl_para, const std::vector<size_t>& tp_ranks, hipStream_t stream);

}  // namespace rtp_llm
