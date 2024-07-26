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

#pragma once

#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/kernels/custom_ar_kernels.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

class CustomAllReduceComm {
public:
    CustomAllReduceComm(const std::vector<int>& tp_ranks, size_t rank);

    ~CustomAllReduceComm();

    void init(const NcclParam& nccl_para, cudaStream_t stream);

    void allReduce(void* output_ptr, size_t elts, DataType data_type, cudaStream_t stream);

    bool checkAllReduceAvailable(size_t elts, DataType data_type);

    static size_t comm_buffer_size() {
        return CUSTOM_AR_SIZE_THRESHOLD;
    }

    static size_t barrier_buffer_size(size_t world_size) {
        return (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t) * world_size * 2;
    }

    static size_t ipc_handle_buffer_size(size_t world_size) {
        return CUDA_IPC_HANDLE_SIZE * world_size;
    }

    static bool shouldCustomAR(const std::vector<int>& tp_ranks, int rank);

    void* peer_comm_buffer_ptr() {
        return param_.peer_comm_buffer_ptrs[rank_];
    }

private:
    std::vector<cudaIpcMemHandle_t> prepareP2PBuffer_(const NcclParam& nccl_para,
                                                      size_t           local_buffer_size,
                                                      void*&           local_buffer_ptr,
                                                      cudaStream_t     stream);

    CustomAllReduceParameters       param_;
    const int                       rank_;
    std::vector<int>                tp_ranks_;
    std::vector<cudaIpcMemHandle_t> peer_comm_buffer_handles_;
};

std::unique_ptr<CustomAllReduceComm> initCustomAllReduceComm(const NcclParam& nccl_para, const std::vector<int>& tp_ranks, cudaStream_t stream);

}  // namespace fastertransformer
