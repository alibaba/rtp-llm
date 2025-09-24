/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "comm_buffer.h"
#include "vec_dtypes.cuh"
#include <cstdint>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace rtp_llm {

__global__ void kuserbuffers_inc(int* id) {
    atomicAdd_system(id, 1);
    // printf("increased id %p from %d to %d\n", id, *id - 1, *id);
}

__global__ void
kuserbuffers_pushrecv(int myrank, int peer, int* recv_id, int* flagptr, int adder, uint64_t ub_timeout) {
    const int signal_id = (*recv_id) + adder;
    *recv_id            = signal_id;
    volatile int* flag  = (volatile int*)flagptr;
    // printf("before pushrecv [nvrank(GPU) dst: %d src: %d]: expecting %d, observe %d from %p\n", myrank, peer,
    // signal_id, *flag, flag);
    if (*flag >= signal_id) {
        return;
    }
    clock_t s = clock64();
    while (CHECK_IDS(*flag, signal_id)) {
        if (CHECK_TIMEOUT(s, ub_timeout)) {
            return;
        }
    }
}

void userbuffers_send(const int     handler,
                      const size_t  srcoffset,
                      const size_t  dstoffset,
                      const size_t  bytes,
                      Communicator* comm,
                      const int     peer,
                      cudaStream_t  stream) {
    int   peerlocal = peer % comm->tp_size;
    void* flagptr   = GET_SEND_PTR_BY_INDEX(peerlocal, comm);
    RTP_LLM_CHECK(INTRANODE(peer));
    void* srcptr = reinterpret_cast<char*>(comm->mem_ptr[handler]) + srcoffset;
    void* dstptr = reinterpret_cast<char*>(comm->peer_ptr[handler][peerlocal]) + dstoffset;
    check_cuda_value(cudaMemcpyAsync(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
    kuserbuffers_inc<<<1, 1, 0, stream>>>(reinterpret_cast<int*>(flagptr));
}

void userbuffers_recv(const int handler, Communicator* comm, const int peer, cudaStream_t stream) {
    int   peerlocal = peer % comm->tp_size;
    void* flagptr   = GET_RECV_PTR_BY_INDEX(peer, comm);

    assert(INTRANODE(peerlocal));
    kuserbuffers_pushrecv<<<1, 1, 0, stream>>>(comm->myrank,
                                               peerlocal,
                                               &comm->recv_id[peer * NVTE_MAX_REGIONS + handler],
                                               reinterpret_cast<int*>(flagptr),
                                               true,
                                               comm->ub_timeout);
}

template<typename T>
__global__ void reduceCudaKernel(
    void* inputs, void* output, const int num_inputs, const int input_size, const size_t aliged_elements_per_input) {
    constexpr uint32_t     vec_size = 16 / sizeof(T);
    vec_t<float, vec_size> first_input_vec, next_input_vec, out_vec;
    const size_t           tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= aliged_elements_per_input) {
        return;
    }

    first_input_vec.cast_load(reinterpret_cast<T*>(inputs) + tid * vec_size);

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
        out_vec[i] = static_cast<float>(first_input_vec[i]);
    }

#pragma unroll
    for (int input_id = 1; input_id < num_inputs; input_id++) {
        next_input_vec.cast_load(reinterpret_cast<T*>(inputs) + input_size * input_id + tid * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            out_vec[i] += static_cast<float>(next_input_vec[i]);
        }
    }

    out_vec.cast_store(reinterpret_cast<T*>(output) + tid * vec_size);
}

template<typename T>
void localReduce(void* inputs, void* output, int num_inputs, int input_size, cudaStream_t stream) {
    constexpr uint32_t vec_size = 16 / sizeof(T);
    RTP_LLM_CHECK(input_size % vec_size == 0);
    size_t    num_threads               = MAX_THREADS / 4;
    const int aliged_elements_per_input = input_size / vec_size;
    size_t    num_blocks                = (aliged_elements_per_input + num_threads - 1) / num_threads;
    dim3      block(num_threads);
    dim3      grid(num_blocks);
    reduceCudaKernel<T><<<grid, block, 0, stream>>>(inputs, output, num_inputs, input_size, aliged_elements_per_input);
}

#define INSTANTIATE_LOCAL_REDUCE_KERNEL(T)                                                                             \
    template void localReduce<T>(void* inputs, void* output, int num_inputs, int input_size, cudaStream_t stream);

#ifdef ENABLE_BF16
INSTANTIATE_LOCAL_REDUCE_KERNEL(__nv_bfloat16)
#endif
INSTANTIATE_LOCAL_REDUCE_KERNEL(half)
INSTANTIATE_LOCAL_REDUCE_KERNEL(float)

}  // namespace rtp_llm