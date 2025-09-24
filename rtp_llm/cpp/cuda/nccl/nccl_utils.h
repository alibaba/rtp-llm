/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#if USING_CUDA
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

#include <cuda_runtime.h>
#include <nccl.h>
#endif
#if USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"

#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include <rccl/rccl.h>
#endif

#include "rtp_llm/cpp/utils/Logger.h"

#include <stdio.h>
#include <string>

#if defined(NCCL_VERSION_CODE) && (NCCL_VERSION_CODE >= 21003)
#define ENABLE_BF16_NCCL
#endif

namespace rtp_llm {
#define NCCLCHECK(cmd)                                                                                                 \
    do {                                                                                                               \
        ncclResult_t __r = cmd;                                                                                        \
        if (__r != ncclSuccess) {                                                                                      \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(__r));                    \
            fflush(stdout);                                                                                            \
            fflush(stderr);                                                                                            \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

struct NcclUid {
    ncclUniqueId nccl_uid_;
    NcclUid() {};
    NcclUid(NcclUid const& uid): nccl_uid_(uid.nccl_uid_) {};
};

struct NcclParam {
    int          rank_{0};
    int          world_size_{1};
    ncclUniqueId nccl_uid_;
    ncclComm_t   nccl_comm_ = nullptr;

    NcclParam(): rank_(0), world_size_(1), nccl_comm_(nullptr) {};
    NcclParam(int rank, int world_size): rank_(rank), world_size_(world_size) {};
    NcclParam(NcclParam const& param):
        rank_(param.rank_), world_size_(param.world_size_), nccl_uid_(param.nccl_uid_), nccl_comm_(param.nccl_comm_) {};
    std::string toString() const {
        return rtp_llm::fmtstr("NcclParam[rank=%d, world_size=%d, nccl_comm=%p]", rank_, world_size_, nccl_comm_);
    }
    bool operator==(const NcclParam& other) const {
        return rank_ == other.rank_ && world_size_ == other.world_size_ && nccl_comm_ == other.nccl_comm_;
    }

    bool operator!=(const NcclParam& other) const {
        return !(*this == other);
    }
};

// New APIs
template<typename T>
void ftNcclAllReduceSum(const T* send_buf, T* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclAllGather(
    const T* send_buf, T* recv_buf, const int data_size, const int rank, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclBroadCast(T* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclRecv(T* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclSend(const T* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

// nccl stream synchronize, abort nccl comms and throw errors when nccl async errors detected
void ftNcclStreamSynchronize(NcclParam tensor_para, cudaStream_t stream, bool timeout = true);

void ftNcclGroupStart();
void ftNcclGroupEnd();
void ftNcclGetUniqueId(NcclUid& uid);
void ftNcclCommInitRank(NcclParam& param, const int rank, const int world_size, const NcclUid uid);
void ftNcclParamDestroy(NcclParam& param);

void ftNcclInitialize(NcclParam&         tensor_para,
                      NcclParam&         pipeline_para,
                      const int          tensor_para_size,
                      const int          pipeline_para_size,
                      const int64_t      world_size,
                      const int64_t      world_rank,
                      const std::string& master_ip,
                      const int          master_port);

void ftNcclInitialize(NcclParam&    tensor_para,
                      NcclParam&    pipeline_para,
                      const int     tensor_para_size,
                      const int     pipeline_para_size,
                      const int64_t world_size,
                      const int64_t world_rank);

std::vector<size_t> fcNcclGatherRanks(NcclParam& para, cudaStream_t stream);

size_t getLocalBatchSize(const size_t batch_size, const size_t seq_len, const size_t pipeline_para_size);

std::vector<int64_t> getLocalParameter(std::vector<int64_t> layer_para, int tensor_para_size);
}  // namespace rtp_llm
