/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include <cuda.h>
#include <tuple>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

namespace user_buffers {

#define NVTE_MAX_REGIONS 16
#define NVTE_MAX_NVLINK 32

// Return TRUE if two ranks share the same NV domain
#define UB_INTRANODE(peer) ((peer / comm->world_size) == (comm->local_rank / comm->world_size))

#define MAX_THREADS 1024

// Return true if producer > consumer, otherwise false while preventing integer overflow
// If we expect that producer will be 2B+ messages behind consumer
#define CHECK_IDS(producer, consumer) (((unsigned)(producer) - (unsigned)(consumer)) & (~INT_MAX))

// Report and error on timeout
#define CHECK_TIMEOUT(t, timeout) ((clock64() - (t)) > timeout)

#define UB_GET_SEND_PTR_BY_INDEX(peerlocal, comm)                                                                      \
    ((reinterpret_cast<char*>((comm)->peer_ptr[0][(peerlocal)])) + (comm->local_rank * sizeof(int)))

#define UB_GET_RECV_PTR_BY_INDEX(recv_peer, comm)                                                                      \
    ((reinterpret_cast<char*>((comm)->mem_ptr[0])) + ((recv_peer) * sizeof(int)))

struct UbCommunicator {
    void*    mem_ptr[NVTE_MAX_REGIONS];   // local rank buffer ptr
    void**   peer_ptr[NVTE_MAX_REGIONS];  // peer rank buffer ptr
    int*     send_id;
    int*     recv_id;
    void*    gpu_ptrs;
    int32_t  local_rank;
    int32_t  world_size;
    uint64_t ub_timeout;
    int      free_region;  // for ub register buffer count
};

void userbuffers_send(const int       handler,
                      const size_t    srcoffset,
                      const size_t    dstoffset,
                      const size_t    bytes,
                      UbCommunicator* comm,
                      const int       peer,
                      cudaStream_t    stream);

void userbuffers_recv(const int handler, UbCommunicator* comm, const int peer, cudaStream_t stream);

void* init_communicator(int64_t local_rank, int64_t world_size);

void destory_communicator(UbCommunicator* comm);

int register_buffer_to_communicator(void* comm_ptr, std::vector<void*> buffer_ptrs);

std::tuple<void*, at::Tensor> allocate_shared_buffer_and_handle(int64_t size);

void* open_mem_handle(at::Tensor& mem_handle);

}  // namespace user_buffers
}  // namespace rtp_llm