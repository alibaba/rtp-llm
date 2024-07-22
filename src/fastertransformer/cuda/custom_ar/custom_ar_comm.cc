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
#include "src/fastertransformer/cuda/cuda_utils.h"
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

namespace fastertransformer {

CustomAllReduceComm::CustomAllReduceComm(const std::vector<int>& tp_ranks, size_t rank)
    : rank_(rank) 
    , tp_ranks_(std::move(tp_ranks))
{
    param_.barrier_flag = 0;
    param_.rank         = rank_;
    param_.local_rank   = rank_;
}

CustomAllReduceComm::~CustomAllReduceComm() {
    // Note: no need to check errors during clean up resource

    // close cudaIPCMemhandle
    for (auto i : tp_ranks_) {
        if (i == rank_) {
            check_cuda_error(cudaFree(param_.peer_comm_buffer_ptrs[i]));
            check_cuda_error(cudaFree(param_.peer_barrier_ptrs[i]));
        } else {
            cudaIpcCloseMemHandle(param_.peer_barrier_ptrs[i]);
            cudaIpcCloseMemHandle(param_.peer_comm_buffer_ptrs[i]);
        }
    }
    // disable P2P access
    for (auto i : tp_ranks_) {
        if (i == rank_) {
            continue;
        } else {
            cudaDeviceDisablePeerAccess(i);
        }
    }
}

bool CustomAllReduceComm::checkAllReduceAvailable(size_t elts, DataType data_type) {
    if (elts * getTypeSize(data_type) > CUSTOM_AR_SIZE_THRESHOLD) {
        return false;
    }
    return true;
}

void CustomAllReduceComm::allReduce(
    void* input_ptr, void* output_ptr, size_t elts, DataType data_type, cudaStream_t stream) {

    param_.elts_total   = elts;
    param_.barrier_flag = FLAG(param_.barrier_flag + 1);
    check_cuda_error(cudaMemcpyAsync(
        param_.peer_comm_buffer_ptrs[rank_], input_ptr, elts * getTypeSize(data_type), cudaMemcpyDeviceToDevice, stream));

    param_.local_output_buffer_ptr = output_ptr;
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, invokeCustomAllReduceDispatch, &param_, stream, tp_ranks_.size());
}

void CustomAllReduceComm::init(const NcclParam& nccl_para, cudaStream_t stream) {
    // enable P2P access
    for (auto i: tp_ranks_) {
        if (i == rank_) {
            continue;
        }
        auto err = cudaDeviceEnablePeerAccess(i, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
            continue;
        }
        check_cuda_error(err);
    }

    // prepare share buffer
    const size_t                    local_comm_buffer_size    = comm_buffer_size();
    const size_t                    local_barrier_buffer_size = barrier_buffer_size(tp_ranks_.size());
    void*                           local_comm_buffer_ptr     = nullptr;
    void*                           local_barrier_buffer_ptr  = nullptr;
    std::vector<cudaIpcMemHandle_t> comm_buffer_handles =
        prepareP2PBuffer_(nccl_para, local_comm_buffer_size, local_comm_buffer_ptr, stream);
    std::vector<cudaIpcMemHandle_t> barrier_buffer_handles =
        prepareP2PBuffer_(nccl_para, local_barrier_buffer_size, local_barrier_buffer_ptr, stream);

    // open other rank's cuda IPCMemHandle
    for (auto i : tp_ranks_) {
        if (i == rank_) {
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
    check_cuda_error(cudaMalloc(&serial_handle_buffer_ptr, ipc_handle_buffer_size(tp_ranks_.size())));

    // open local cudaIpcMemHandle
    cudaIpcMemHandle_t local_buffer_handle;
    check_cuda_error(cudaIpcGetMemHandle(&local_buffer_handle, local_buffer_ptr));

    // serialized cudaIpcMemHandle
    check_cuda_error(cudaMemcpyAsync(serial_handle_buffer_ptr + CUDA_IPC_HANDLE_SIZE * rank_,
                                     local_buffer_handle.reserved,
                                     CUDA_IPC_HANDLE_SIZE,
                                     cudaMemcpyHostToDevice,
                                     stream));

    // all gather serialized cudaIpcMemHandle
    ftNcclAllGather(serial_handle_buffer_ptr, serial_handle_buffer_ptr, CUDA_IPC_HANDLE_SIZE, rank_, nccl_para, stream);
    check_cuda_error(cudaStreamSynchronize(stream));

    // deserialize all ranks' cudaIpcMemHandle
    std::vector<cudaIpcMemHandle_t> handles(tp_ranks_.size());
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

    size_t world_size = tp_ranks.size();
    char* disable_custom_ar_str = std::getenv("FT_DISABLE_CUSTOM_AR");
    bool  disable_custom_ar = disable_custom_ar_str != nullptr;
    if (disable_custom_ar) {
        FT_LOG_INFO("Disable custom ar since FT_DISABLE_CUSTOM_AR is set");
        return false;
    }

    // TODO(xyz): check whether nvlink is enabled and all ranks are all same nodes

    // check whether world size is valid
    std::unordered_set<int> available_world_sizes = {2, 4, 8};
    if (available_world_sizes.find(world_size) == available_world_sizes.end()) {
        FT_LOG_INFO("Disable custom ar for invalid world size %d", world_size);
        return false;
    }

    // check P2P access
    for (auto i : tp_ranks) {
        if (i == rank) {
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

std::unique_ptr<CustomAllReduceComm> initCustomAllReduceComm(const NcclParam& nccl_para, const std::vector<int>& tp_ranks, cudaStream_t stream) {
    if (!CustomAllReduceComm::shouldCustomAR(tp_ranks, nccl_para.rank_)) {
        return nullptr;
    }
    auto comm = std::make_unique<CustomAllReduceComm>(tp_ranks, nccl_para.rank_);
    comm->init(nccl_para, stream);
    FT_LOG_INFO("Custom all reduce is enabled on rank %d of %d", nccl_para.rank_, tp_ranks.size());
    return comm;
}

}  // namespace fastertransformer
