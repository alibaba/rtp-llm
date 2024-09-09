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

#include "src/fastertransformer/core/Types.h"
#if USING_CUDA
#include "src/fastertransformer/cuda/cuda_utils.h"
#elif USING_ROCM
#include "src/fastertransformer/rocm/hip_utils.h"
#include "src/fastertransformer/rocm/cuda_shims.h"
#endif
#include "src/fastertransformer/cuda/memory_utils.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/utils/logger.h"
#include <climits>
#include <cstdint>
#include <sys/types.h>
#include <iostream>
#include <unordered_set>
#include <vector>
using namespace std;

#if USING_CUDA
#define getVisibleDevices fastertransformer::getVisibleCUDADevices
#elif USING_ROCM
#define getVisibleDevices fastertransformer::rocm::getVisibleCUDADevices
#endif

namespace fastertransformer {

CustomAllReduceComm::CustomAllReduceComm(const std::vector<int>& tp_ranks, size_t rank):
    rank_(rank), world_size_(tp_ranks.size()), tp_ranks_(std::move(tp_ranks)) {
    param_.barrier_flag = 0;
    param_.rank         = rank_;
    param_.local_rank   = rank_;

    for (size_t i = 0; i < world_size_; i++) {
        if (tp_ranks[i] == rank) {
            rank_index_ = i;
            break;
        }
    }
}

CustomAllReduceComm::~CustomAllReduceComm() {
    // close cudaIPCMemhandle
    for (size_t i = 0; i < world_size_; i++) {
        if (i == rank_index_) {
            check_cuda_error(cudaFree(param_.peer_comm_buffer_ptrs[i]));
            check_cuda_error(cudaFree(param_.peer_barrier_ptrs[i]));
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

bool CustomAllReduceComm::checkAllReduceAvailable(size_t elts, DataType data_type) {
    if (elts * getTypeSize(data_type) > CUSTOM_AR_SIZE_THRESHOLD
        || elts % (MAX_RANKS_PER_NODE * MAX_RANKS_PER_NODE) != 0) {
        return false;
    }
    return true;
}

void CustomAllReduceComm::allReduce(
    void* input_ptr, void* output_ptr, size_t elts, DataType data_type, cudaStream_t stream) {
    if (input_ptr != peer_comm_buffer_ptr()) {
        std::string err_msg =
            "input_ptr != peer_comm_buffer_ptr, check whether BufferPtr after prepareAllReduce is released or replaced";
        FT_LOG_INFO(err_msg);
        throw std::runtime_error(err_msg);
    }
    param_.elts_total              = elts;
    param_.barrier_flag            = FLAG(param_.barrier_flag + 1);
    param_.local_output_buffer_ptr = output_ptr;
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeCustomAllReduceDispatch, &param_, stream, world_size_);
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
        check_cuda_error(err);
    }

    // prepare share buffer
    const size_t                    local_comm_buffer_size    = comm_buffer_size();
    const size_t                    local_barrier_buffer_size = barrier_buffer_size(world_size_);
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
            check_cuda_error(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&comm_foreign_buffer),
                                                  comm_buffer_handles[i],
                                                  cudaIpcMemLazyEnablePeerAccess));
            param_.peer_comm_buffer_ptrs[i] = reinterpret_cast<void*>(comm_foreign_buffer);

            uint8_t* barrier_forier_buffer{nullptr};
            check_cuda_error(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&barrier_forier_buffer),
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
    check_cuda_error(cudaMalloc(&local_buffer_ptr, local_buffer_size));
    check_cuda_error(cudaMemset(local_buffer_ptr, 0, local_buffer_size));

    // malloc serial handle buffer
    char* serial_handle_buffer_ptr;
    check_cuda_error(cudaMalloc(&serial_handle_buffer_ptr, ipc_handle_buffer_size(world_size_)));

    // open local cudaIpcMemHandle
    cudaIpcMemHandle_t local_buffer_handle;
    check_cuda_error(cudaIpcGetMemHandle(&local_buffer_handle, local_buffer_ptr));

    // serialized cudaIpcMemHandle
    check_cuda_error(cudaMemcpyAsync(serial_handle_buffer_ptr + CUDA_IPC_HANDLE_SIZE * rank_index_,
                                     local_buffer_handle.reserved,
                                     CUDA_IPC_HANDLE_SIZE,
                                     cudaMemcpyHostToDevice,
                                     stream));

    // all gather serialized cudaIpcMemHandle
    ftNcclAllGather(
        serial_handle_buffer_ptr, serial_handle_buffer_ptr, CUDA_IPC_HANDLE_SIZE, rank_index_, nccl_para, stream);
    check_cuda_error(cudaStreamSynchronize(stream));

    // deserialize all ranks' cudaIpcMemHandle
    std::vector<cudaIpcMemHandle_t> handles(world_size_);
    for (size_t i = 0; i < handles.size(); ++i) {
        check_cuda_error(cudaMemcpyAsync(handles[i].reserved,
                                         serial_handle_buffer_ptr + CUDA_IPC_HANDLE_SIZE * i,
                                         CUDA_IPC_HANDLE_SIZE,
                                         cudaMemcpyDeviceToHost,
                                         stream));
    }

    check_cuda_error(cudaFreeAsync(serial_handle_buffer_ptr, stream));
    return handles;
}

bool CustomAllReduceComm::shouldCustomAR(const std::vector<int>& tp_ranks, int rank) {

    size_t world_size       = tp_ranks.size();
    size_t local_world_size = getVisibleDevices().size();
    // check whether all ranks are on same nodes
    if (world_size != local_world_size) {
        FT_LOG_INFO("Disable custom ar since TP is performanced on multi nodes, world_size=%d, local_world_size=%d",
                    world_size,
                    local_world_size);
        return false;
    }

    char*  disable_custom_ar_str = std::getenv("FT_DISABLE_CUSTOM_AR");
    bool   disable_custom_ar     = disable_custom_ar_str != nullptr && std::string(disable_custom_ar_str) == "1";
    if (disable_custom_ar) {
        FT_LOG_INFO("Disable custom ar since FT_DISABLE_CUSTOM_AR is set");
        return false;
    }

    // check whether world size is valid
    std::unordered_set<int> available_world_sizes = {2, 4, 8};
    if (available_world_sizes.find(world_size) == available_world_sizes.end()) {
        FT_LOG_INFO("Disable custom ar for invalid world size %d", world_size);
        return false;
    }

#if USING_CUDA
    std::string driver_version     = getDriverVersion();
    bool        driver_version_535 = driver_version.size() >= 3 && driver_version.substr(0, 3) == "535";
    int         cuda_version       = getCudaVersion();
    FT_LOG_INFO("Nvidia driver version: %s, Cuda version: %d", driver_version.c_str(), cuda_version);
    if (cuda_version <= 12040 && driver_version_535 && !checkAllNVLinks(tp_ranks)) {
        if (!checkOnSameNumaNodes(tp_ranks)) {
            FT_LOG_INFO(
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
        check_cuda_error(cudaDeviceCanAccessPeer(&peer_access_available, rank, i));
        if (peer_access_available == 0) {
            FT_LOG_INFO("Disable custom all reduce since device %d and device %d do not have peer access", rank, i);
            return false;
        }
    }

    return true;
}

std::unique_ptr<CustomAllReduceComm>
initCustomAllReduceComm(const NcclParam& nccl_para, const std::vector<int>& tp_ranks, cudaStream_t stream) {
    if (!CustomAllReduceComm::shouldCustomAR(tp_ranks, nccl_para.rank_)) {
        return nullptr;
    }
    auto comm = std::make_unique<CustomAllReduceComm>(tp_ranks, nccl_para.rank_);
    comm->init(nccl_para, stream);
    FT_LOG_INFO("Custom all reduce is enabled on rank %d of %d", nccl_para.rank_, tp_ranks.size());
    return comm;
}

}  // namespace fastertransformer
