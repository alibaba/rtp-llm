/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "custom_ar_comm.h"

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <climits>
#include <cstdint>
#include <sys/types.h>
#include <iostream>
#include <unordered_set>
#include <vector>
using namespace std;

namespace rtp_llm {

CustomAllReduceComm::CustomAllReduceComm(const std::vector<size_t>& tp_ranks, size_t rank, size_t rank_index):
    rank_(rank),
    rank_index_(rank_index),
    world_size_(tp_ranks.size()),
    support_nv_link_(checkAllNVLinks(tp_ranks)),
    comm_buf_threshold_(getCommBufThreshold(support_nv_link_, world_size_)),
    tp_ranks_(std::move(tp_ranks)) {
    param_.barrier_flag        = 0;
    param_.rank                = rank_;
    param_.local_rank          = rank_;
    param_.max_elts_total_size = comm_buf_threshold_;
    param_.ranks_per_node      = world_size_;
}

CustomAllReduceComm::~CustomAllReduceComm() {
    // close cudaIPCMemhandle
    for (size_t i = 0; i < world_size_; i++) {
        if (i == rank_index_) {
            check_cuda_value(cudaFree(param_.peer_comm_buffer_ptrs[i]));
            check_cuda_value(cudaFree(param_.peer_barrier_ptrs[i]));
        } else {
            cudaIpcCloseMemHandle(param_.peer_barrier_ptrs[i]);
            cudaIpcCloseMemHandle(param_.peer_comm_buffer_ptrs[i]);
        }
    }
    // disable P2P access
    for (size_t i = 0; i < world_size_; i++) {
        if (i == rank_index_) {
            continue;
        } else {
            cudaDeviceDisablePeerAccess(tp_ranks_[i]);
        }
    }
}

bool CustomAllReduceComm::checkAllReduceAvailable(size_t elts_total_num, DataType data_type, size_t world_size) {
    size_t elts_total_size = elts_total_num * getTypeSize(data_type);
    if (elts_total_size >= comm_buf_threshold_) {
        return false;
    }

    size_t                elts_per_thread = 16 / getTypeSize(data_type);
    AllReduceStrategyType algo            = select_kernel_algo(support_nv_link_, elts_total_size, world_size);
    size_t const msg_align = (algo == AllReduceStrategyType::TWOSHOT) ? world_size * elts_per_thread : elts_per_thread;
    // RTP_LLM_LOG_INFO("elts_num :%d, algo : %d, msg_align %d, elts_per_thread %d", elts_total_num, algo, msg_align,
    // elts_per_thread);
    return elts_total_num % msg_align == 0;
}

void CustomAllReduceComm::allReduce(
    void* input_ptr, void* output_ptr, size_t elts_total_num, DataType data_type, cudaStream_t stream) {
    if (input_ptr != peerCommBufferPtr()) {
        std::string err_msg =
            "input_ptr != peer_comm_buffer_ptr, check whether BufferPtr after prepareAllReduce is released or replaced";
        RTP_LLM_LOG_INFO(err_msg);
        throw std::runtime_error(err_msg);
    }
    param_.elts_total_num          = elts_total_num;
    param_.elts_per_rank           = world_size_;
    param_.data_type_size          = getTypeSize(data_type);
    param_.barrier_flag            = FLAG(param_.barrier_flag + 1);
    param_.local_output_buffer_ptr = output_ptr;
    param_.kernel_algo =
        select_kernel_algo(support_nv_link_, param_.elts_total_num * param_.data_type_size, world_size_);
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeCustomAllReduceDispatch, &param_, param_.barrier_flag, stream);
}

void CustomAllReduceComm::init(const NcclParam& nccl_para, cudaStream_t stream) {
    // enable P2P access
    for (size_t i = 0; i < world_size_; i++) {
        if (i == rank_index_) {
            continue;
        }
        auto err = cudaDeviceEnablePeerAccess(tp_ranks_[i], 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
            continue;
        }
        check_cuda_value(err);
    }

    // prepare share buffer
    const size_t                    local_comm_buffer_size    = comm_buf_threshold_;
    const size_t                    local_barrier_buffer_size = barrierBufSize(world_size_);
    void*                           local_comm_buffer_ptr     = nullptr;
    void*                           local_barrier_buffer_ptr  = nullptr;
    std::vector<cudaIpcMemHandle_t> comm_buffer_handles =
        prepareP2PBuffer_(nccl_para, local_comm_buffer_size, local_comm_buffer_ptr, stream);
    std::vector<cudaIpcMemHandle_t> barrier_buffer_handles =
        prepareP2PBuffer_(nccl_para, local_barrier_buffer_size, local_barrier_buffer_ptr, stream);

    // open other rank's cuda IPCMemHandle
    for (size_t i = 0; i < world_size_; i++) {
        if (i == rank_index_) {
            param_.peer_comm_buffer_ptrs[i] = local_comm_buffer_ptr;
            param_.peer_barrier_ptrs[i]     = reinterpret_cast<uint32_t*>(local_barrier_buffer_ptr);
        } else {
            uint8_t* comm_foreign_buffer{nullptr};
            check_cuda_value(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&comm_foreign_buffer),
                                                  comm_buffer_handles[i],
                                                  cudaIpcMemLazyEnablePeerAccess));
            param_.peer_comm_buffer_ptrs[i] = reinterpret_cast<void*>(comm_foreign_buffer);

            uint8_t* barrier_forier_buffer{nullptr};
            check_cuda_value(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&barrier_forier_buffer),
                                                  barrier_buffer_handles[i],
                                                  cudaIpcMemLazyEnablePeerAccess));
            param_.peer_barrier_ptrs[i] = reinterpret_cast<uint32_t*>(barrier_forier_buffer);
        }
    }
}

std::vector<cudaIpcMemHandle_t> CustomAllReduceComm::prepareP2PBuffer_(const NcclParam& nccl_para,
                                                                       size_t           local_buffer_size,
                                                                       void*&           local_buffer_ptr,
                                                                       cudaStream_t     stream) {
    // Note: The existing function cudaIpcGetMemHandle works only with memory allocated through
    // cudaMalloc and cannot be used on any memory allocated through cudaMallocAsync, regardless of whether the memory
    // was allocated from an explicit pool. Therefore, we can not use buffer_manager to allocate the memory

    // see https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/

    // malloc and reset local buffer
    check_cuda_value(cudaMalloc(&local_buffer_ptr, local_buffer_size));
    check_cuda_value(cudaMemset(local_buffer_ptr, 0, local_buffer_size));

    // malloc serial handle buffer
    char* serial_handle_buffer_ptr;
    check_cuda_value(cudaMalloc(&serial_handle_buffer_ptr, IPChandleBufSize(world_size_)));

    // open local cudaIpcMemHandle
    cudaIpcMemHandle_t local_buffer_handle;
    check_cuda_value(cudaIpcGetMemHandle(&local_buffer_handle, local_buffer_ptr));

    // serialized cudaIpcMemHandle
    check_cuda_value(cudaMemcpyAsync(serial_handle_buffer_ptr + CUDA_IPC_HANDLE_SIZE * rank_index_,
                                     local_buffer_handle.reserved,
                                     CUDA_IPC_HANDLE_SIZE,
                                     cudaMemcpyHostToDevice,
                                     stream));

    // all gather serialized cudaIpcMemHandle
    ftNcclAllGather(
        serial_handle_buffer_ptr, serial_handle_buffer_ptr, CUDA_IPC_HANDLE_SIZE, rank_index_, nccl_para, stream);
    check_cuda_value(cudaStreamSynchronize(stream));

    // deserialize all ranks' cudaIpcMemHandle
    std::vector<cudaIpcMemHandle_t> handles(world_size_);
    for (size_t i = 0; i < handles.size(); ++i) {
        check_cuda_value(cudaMemcpyAsync(handles[i].reserved,
                                         serial_handle_buffer_ptr + CUDA_IPC_HANDLE_SIZE * i,
                                         CUDA_IPC_HANDLE_SIZE,
                                         cudaMemcpyDeviceToHost,
                                         stream));
    }

    check_cuda_value(cudaFreeAsync(serial_handle_buffer_ptr, stream));
    return handles;
}

bool CustomAllReduceComm::shouldCustomAR(const std::vector<size_t>& tp_ranks,
                                         size_t                     rank,
                                         const HWKernelConfig&      hw_kernel_config) {

    size_t world_size       = tp_ranks.size();
    size_t local_world_size = getVisibleDeviceNum();
    // check whether all ranks are on same nodes
    if (world_size != local_world_size) {
        RTP_LLM_LOG_INFO(
            "Disable custom ar since TP is performanced on multi nodes, world_size=%d, local_world_size=%d",
            world_size,
            local_world_size);
        return false;
    }

    if (hw_kernel_config.ft_disable_custom_ar) {
        RTP_LLM_LOG_INFO("Disable custom ar since FT_DISABLE_CUSTOM_AR is set");
        return false;
    }

    // check whether world size is valid
    std::unordered_set<size_t> available_world_sizes = {2, 4, 8, 16};
    if (available_world_sizes.find(world_size) == available_world_sizes.end()) {
        RTP_LLM_LOG_INFO("Disable custom ar for invalid world size %d", world_size);
        return false;
    }

#if USING_CUDA
    std::string driver_version     = getDriverVersion();
    bool        driver_version_535 = driver_version.size() >= 3 && driver_version.substr(0, 3) == "535";
    int         cuda_version       = getCudaVersion();
    RTP_LLM_LOG_INFO("Nvidia driver version: %s, Cuda version: %d", driver_version.c_str(), cuda_version);
    if (cuda_version <= 12040 && driver_version_535 && !checkAllNVLinks(tp_ranks)) {
        if (!checkOnSameNumaNodes(tp_ranks)) {
            RTP_LLM_LOG_INFO(
                "Disable custom ar since since there exists bug for p2p comm across device accross numa node when nvidia driver version: %s, Cuda version: %d",
                driver_version.c_str(),
                cuda_version);
            return false;
        }
    }
#endif

    // check P2P access
    for (size_t i = 0; i < tp_ranks.size(); i++) {
        size_t peer_rank = tp_ranks[i];
        if (peer_rank == rank) {
            continue;
        }
        int peer_access_available = 0;
        check_cuda_value(cudaDeviceCanAccessPeer(&peer_access_available, rank, i));
        if (peer_access_available == 0) {
            RTP_LLM_LOG_INFO(
                "Disable custom all reduce since device %d and device %d do not have peer access", rank, i);
            return false;
        }
    }

    return true;
}

size_t CustomAllReduceComm::getCommBufThreshold(bool support_nv_link, size_t world_size) {
    size_t custom_ar_size_threshold = 0;
    if (world_size <= 2) {
        if (support_nv_link) {
            custom_ar_size_threshold = 16 * 1000 * 1000;
        } else {
            custom_ar_size_threshold = 2 * 1000 * 1000;
        }
    } else {
        constexpr size_t elts_per_thread = 8;
        size_t           mod     = elts_per_thread * world_size * DEFAULT_BLOCK_SIZE * MAX_ALL_REDUCE_BLOCKS * 2;
        custom_ar_size_threshold = 8 * 1000 * 1000;
        custom_ar_size_threshold +=
            (custom_ar_size_threshold % mod) == 0 ? 0 : (mod - (custom_ar_size_threshold % mod));
    }
    return custom_ar_size_threshold;
}

AllReduceStrategyType
CustomAllReduceComm::select_kernel_algo(bool support_nv_link, size_t elts_total_size, size_t ranks_per_node) const {
    AllReduceStrategyType kernel_algo;
    if (ranks_per_node <= 2) {
        kernel_algo = AllReduceStrategyType::ONESHOT;
    } else if (ranks_per_node <= 4) {
        if (support_nv_link) {
            kernel_algo =
                elts_total_size < 1 * 1000 * 1000 ? AllReduceStrategyType::ONESHOT : AllReduceStrategyType::TWOSHOT;
        } else {
            kernel_algo =
                elts_total_size < 500 * 1000 ? AllReduceStrategyType::ONESHOT : AllReduceStrategyType::TWOSHOT;
        }
    } else {
        if (support_nv_link) {
            kernel_algo =
                elts_total_size < 500 * 1000 ? AllReduceStrategyType::ONESHOT : AllReduceStrategyType::TWOSHOT;
        } else {
            kernel_algo =
                elts_total_size < 250 * 1000 ? AllReduceStrategyType::ONESHOT : AllReduceStrategyType::TWOSHOT;
        }
    }
    return kernel_algo;
}

std::unique_ptr<CustomAllReduceComm> initCustomAllReduceComm(const NcclParam&           nccl_para,
                                                             const std::vector<size_t>& tp_ranks,
                                                             cudaStream_t               stream,
                                                             const HWKernelConfig&      hw_kernel_config) {
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

    auto comm = std::make_unique<CustomAllReduceComm>(tp_ranks, nccl_para.rank_, rank_index);
    comm->init(nccl_para, stream);
    RTP_LLM_LOG_INFO("Custom all reduce is enabled on rank %d of %d", nccl_para.rank_, tp_ranks.size());
    return comm;
}

}  // namespace rtp_llm
