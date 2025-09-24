/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <mutex>

#include "rtp_llm/cpp/kernels/comm_buffer.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/cuda/custom_ar/custom_ar_comm.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"

#define LOCALSIZE2 4 * NVTE_MAX_NVLINK
#define NUM_MAX_STREAM 3

namespace rtp_llm {

int register_user_buffer_collective(
    void** gpubuff, size_t bytes, Communicator* comm, const NcclParam& nccl_para, cudaStream_t stream);
/*  returns handler and registers buffers. assumed to be collective i.e. you use same groups and
   dont mix buffers for different operations returns -1 if cant register (too many preregistered
   regions already) if alloc==true will allocate memory and fill the pointers (required for NVL
   SHARP and NSO/MNNVL)
*/

struct CommBuffer {
    ~CommBuffer();

    CommBuffer(const NcclParam&           nccl_para,
               const std::vector<size_t>& buffer_shape,
               DataType                   buffer_dtype,
               const std::vector<size_t>& tp_ranks,
               size_t                     myrank,
               bool                       is_ag,
               cudaStream_t               stream);

    Communicator* _comm = nullptr;
    int           _ub_reg;
    int           _gemm_priority;
    int           _comm_priority;
    bool          _is_reduce_scatter{false};
    int           _num_ubuf_chunks;

    int                       _next_rank;
    int                       _prev_rank;
    std::vector<void*>        _ubufs;
    cudaEvent_t               _stop_send, _stop_recv;
    void*                     _ubuf;
    std::vector<cudaStream_t> _stream_send;
    cudaStream_t              _stream_recv;
    std::vector<cudaStream_t> _stream_compute;
    cudaEvent_t               _start_compute, _stop_compute, _start_comm;
    size_t                    _rank_chunk_stride;
};

std::unique_ptr<CommBuffer> initCommBuffer(std::vector<size_t>&       buffer_shape,
                                           DataType                   buffer_type,
                                           const NcclParam&           nccl_para,
                                           const std::vector<size_t>& tp_ranks,
                                           bool                       is_ag,
                                           cudaStream_t               stream);

void invokeLocalReduceDispatch(
    DataType data_type, void* inputs, void* output, int num_inputs, int input_size, cudaStream_t stream);

}  // namespace rtp_llm
