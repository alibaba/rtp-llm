/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include <assert.h>
#include <vector>
#include <stdint.h>

#if USING_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#define NVTE_MAX_REGIONS 16
#define NVTE_MAX_NVLINK 32

// Return TRUE if two ranks share the same NV domain
#define INTRANODE(peer) ((peer / comm->tp_size) == (comm->myrank / comm->tp_size))

#define MAX_THREADS 1024

#define GET_SEND_PTR_BY_INDEX(peerlocal, comm)                                                                         \
    ((reinterpret_cast<char*>((comm)->peer_ptr[0][(peerlocal)])) + (comm->myrank * sizeof(int)))

#define GET_RECV_PTR_BY_INDEX(recv_peer, comm)                                                                         \
    ((reinterpret_cast<char*>((comm)->mem_ptr[0])) + ((recv_peer) * sizeof(int)))

// Return true if producer > consumer, otherwise false while preventing integer overflow
// If we expect that producer will be 2B+ messages behind consumer
#define CHECK_IDS(producer, consumer) (((unsigned)(producer) - (unsigned)(consumer)) & (~INT_MAX))

// Report and error on timeout
#define CHECK_TIMEOUT(t, timeout) ((clock64() - (t)) > timeout)

#define CHECK_CE(ce_start, ce_end) ((ce_start) != nullptr && (ce_end) != nullptr && *(ce_start) != *(ce_end))

namespace rtp_llm {

struct Communicator {
    std::vector<size_t> tp_ranks;
    size_t              tp_size = 0;
    size_t              myrank  = 0;
    int                 free_region;
    void*               gpu_ptrs;
    void*               mem_ptr[NVTE_MAX_REGIONS];
    void**              peer_ptr[NVTE_MAX_REGIONS];
    int*                recv_id;
    uint64_t            ub_timeout;
};

void userbuffers_send(const int     handler,
                      const size_t  srcoffset,
                      const size_t  dstoffset,
                      const size_t  bytes,
                      Communicator* comm,
                      const int     peer,
                      cudaStream_t  stream = 0);

void userbuffers_recv(const int handler, Communicator* comm, const int peer, cudaStream_t stream = 0);

template<typename T>
void localReduce(void* input, void* output, int num_inputs, int input_size, cudaStream_t stream);

}  // namespace rtp_llm
