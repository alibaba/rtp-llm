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

#include <cstddef>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/kernels/custom_ar_kernels.h"

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class CustomAllReduceComm {
public:
    CustomAllReduceComm(const std::vector<size_t>& tp_ranks, size_t rank, size_t rank_index);

    ~CustomAllReduceComm();

    void init(const NcclParam& nccl_para, cudaStream_t stream);

    void allReduce(void* input_ptr, void* output_ptr, size_t elts_total_num, DataType data_type, cudaStream_t stream);

    bool checkAllReduceAvailable(size_t elts_total_num, DataType data_type, size_t world_size);

    static bool
    shouldCustomAR(const std::vector<size_t>& tp_ranks, size_t rank, const HWKernelConfig& hw_kernel_config);

    void* peerCommBufferPtr() {
        return param_.peer_comm_buffer_ptrs[rank_];
    }

private:
    static size_t getCommBufThreshold(bool support_nv_link, size_t world_size);

    size_t barrierBufSize(size_t world_size) const {
        return (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t) * world_size * 3;
    }

    size_t IPChandleBufSize(size_t world_size) const {
        return CUDA_IPC_HANDLE_SIZE * world_size;
    }

    AllReduceStrategyType select_kernel_algo(bool support_nv_link, size_t elts_total_size, size_t ranks_per_node) const;
    std::vector<cudaIpcMemHandle_t> prepareP2PBuffer_(const NcclParam& nccl_para,
                                                      size_t           local_buffer_size,
                                                      void*&           local_buffer_ptr,
                                                      cudaStream_t     stream);

    CustomAllReduceParameters       param_;
    const size_t                    rank_               = 0;
    const size_t                    rank_index_         = 0;
    const size_t                    world_size_         = 0;
    bool                            support_nv_link_    = false;
    const size_t                    comm_buf_threshold_ = 0;
    std::vector<size_t>             tp_ranks_;
    std::vector<cudaIpcMemHandle_t> peer_comm_buffer_handles_;
};

std::unique_ptr<CustomAllReduceComm> initCustomAllReduceComm(const NcclParam&           nccl_para,
                                                             const std::vector<size_t>& tp_ranks,
                                                             cudaStream_t               stream,
                                                             const HWKernelConfig&      hw_kernel_config);

}  // namespace rtp_llm
