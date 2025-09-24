/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "comm_buffer.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"

namespace rtp_llm {

CommBuffer::CommBuffer(const NcclParam&           nccl_para,
                       const std::vector<size_t>& buffer_shape,
                       DataType                   buffer_dtype,
                       const std::vector<size_t>& tp_ranks,
                       size_t                     myrank,
                       bool                       is_ag,
                       cudaStream_t               stream) {
    RTP_LLM_LOG_INFO("Create Userbuffers Communicator\n");

    _comm              = new Communicator();
    Communicator* comm = _comm;
    comm->tp_ranks     = tp_ranks;
    comm->tp_size      = tp_ranks.size();
    comm->myrank       = myrank;
    comm->free_region  = 0;

    int device_clock = 0;
    // 110 sec wait time by default
    int sec_timeout = 110;
    int cur_dev;
    cur_dev = getDevice();
    check_cuda_value(cudaDeviceGetAttribute(&device_clock, cudaDevAttrClockRate, cur_dev));
    comm->ub_timeout = 1000ull * device_clock * sec_timeout;
    RTP_LLM_LOG_DEBUG(
        "UB_TIMEOUT is set to %d sec, %" PRIu64 " cycles, freq: %dkhz\n", sec_timeout, comm->ub_timeout, device_clock);

    // peer pointers + op flags + comm buffer
    register_user_buffer_collective(&(comm->gpu_ptrs), LOCALSIZE2, _comm, nccl_para, stream);
    check_cuda_value(cudaMalloc(&comm->recv_id, NVTE_MAX_REGIONS * comm->tp_size * sizeof(int)));
    check_cuda_value(cudaMemset(comm->recv_id, 0, NVTE_MAX_REGIONS * comm->tp_size * sizeof(int)));
    check_cuda_value(cudaDeviceSynchronize());

    priorityRange(&_gemm_priority, &_comm_priority);
    size_t tp_size = tp_ranks.size();
    for (int i = 0; i < std::min((size_t)NUM_MAX_STREAM, tp_size); i++) {
        cudaStream_t stream;
        check_cuda_value(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, _gemm_priority));
        _stream_compute.push_back(std::move(stream));
    }

    // CUDA event creation
    check_cuda_value(cudaEventCreateWithFlags(&_start_compute, 0));
    check_cuda_value(cudaEventCreateWithFlags(&_stop_compute, 0));
    check_cuda_value(cudaEventCreateWithFlags(&_start_comm, 0));

    _is_reduce_scatter = !is_ag;

    // Create workspace tensor with userbuffer
    RTP_LLM_CHECK_WITH_INFO(buffer_shape.size() == 2, "Userbuffer shape must be 2-dimensional!");
    size_t buffer_bytes       = buffer_shape[0] * buffer_shape[1] * getTypeSize(buffer_dtype);
    int    buffer_chunk_bytes = buffer_bytes / tp_size;
    _rank_chunk_stride        = buffer_chunk_bytes;
    _num_ubuf_chunks          = tp_size;
    if (_is_reduce_scatter) {
        // GEMM + RS overlap: Allocate `2 x tp_size - 1` buffers to hold recieved GEMM chunk
        // outputs for reduction at the end of the pipelining.
        buffer_bytes     = buffer_bytes / tp_size * (tp_size * 2 - 1);
        _num_ubuf_chunks = tp_size * 2 - 1;
    }
    RTP_LLM_LOG_DEBUG(
        "buffer_shape[0] %zu, buffer_shape[1] %zu, getTypeSize(buffer_dtype) %zu, buffer_bytes %zu, tp_size %d",
        buffer_shape[0],
        buffer_shape[1],
        getTypeSize(buffer_dtype),
        buffer_bytes,
        tp_size);

    void* buffer_ptr;
    _ub_reg = register_user_buffer_collective(&buffer_ptr, buffer_bytes, _comm, nccl_para, stream);
    _ubuf   = buffer_ptr;

    // Create tensor chunks for easy management
    char* ubuf_byte_ptr = reinterpret_cast<char*>(buffer_ptr);
    for (int i = 0; i < _num_ubuf_chunks; i++) {
        _ubufs.push_back(ubuf_byte_ptr);
        ubuf_byte_ptr += buffer_chunk_bytes;
    }

    _next_rank = (tp_size + myrank + 1) % tp_size;
    _prev_rank = (tp_size + myrank + -1) % tp_size;

    for (int i = 0; i < std::min((size_t)NUM_MAX_STREAM, tp_size); i++) {
        cudaStream_t stream;
        check_cuda_value(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, _comm_priority));
        _stream_send.push_back(std::move(stream));
    }
    check_cuda_value(cudaStreamCreateWithPriority(&_stream_recv, cudaStreamNonBlocking, _comm_priority));
    check_cuda_value(cudaEventCreateWithFlags(&_stop_send, 0));
    check_cuda_value(cudaEventCreateWithFlags(&_stop_recv, 0));
}

CommBuffer::~CommBuffer() {
    for (int hndl = 0; hndl < _comm->free_region; hndl++) {
        for (int rank = 0; rank < _comm->tp_size; rank++) {
            if (rank != _comm->myrank) {
                cudaIpcCloseMemHandle(_comm->peer_ptr[hndl][rank]);
            } else {
                check_cuda_value(cudaFree(reinterpret_cast<void*>(_comm->peer_ptr[hndl][rank])));
            }
        }
        free(_comm->peer_ptr[hndl]);
    }
    check_cuda_value(cudaFree(reinterpret_cast<void*>(_comm->recv_id)));
    check_cuda_value(cudaEventDestroy(_stop_recv));
    check_cuda_value(cudaEventDestroy(_stop_send));
    check_cuda_value(cudaEventDestroy(_start_compute));
    check_cuda_value(cudaEventDestroy(_stop_compute));
    check_cuda_value(cudaEventDestroy(_start_comm));
    delete _comm;
}

int register_user_buffer_collective(
    void** gpubuff, size_t bytes, Communicator* comm, const NcclParam& nccl_para, cudaStream_t stream) {
    if (comm->free_region > NVTE_MAX_REGIONS)
        return -1;
    int hndl             = comm->free_region;
    comm->peer_ptr[hndl] = reinterpret_cast<void**>(malloc(sizeof(void*) * (comm->tp_size)));

    check_cuda_value(cudaMalloc(gpubuff, bytes));
    check_cuda_value(cudaMemset(*gpubuff, 0, bytes));

    RTP_LLM_CHECK_WITH_INFO(comm->tp_size <= 8, "CUDA IPC supports only up to 8 GPUs in an NVLink domain.");
    cudaIpcMemHandle_t memhndl;
    check_cuda_value(cudaIpcGetMemHandle(&memhndl, *gpubuff));

    char* serial_handle_buffer_ptr;
    check_cuda_value(cudaMalloc(&serial_handle_buffer_ptr, comm->tp_size * CUDA_IPC_HANDLE_SIZE));

    // serialized cudaIpcMemHandle
    check_cuda_value(cudaMemcpyAsync(serial_handle_buffer_ptr + CUDA_IPC_HANDLE_SIZE * comm->myrank,
                                     memhndl.reserved,
                                     CUDA_IPC_HANDLE_SIZE,
                                     cudaMemcpyHostToDevice,
                                     stream));

    ftNcclAllGather(reinterpret_cast<char*>(serial_handle_buffer_ptr),
                    reinterpret_cast<char*>(serial_handle_buffer_ptr),
                    CUDA_IPC_HANDLE_SIZE,
                    comm->myrank,
                    nccl_para,
                    stream);
    RTP_LLM_LOG_DEBUG("nccl_param %s", nccl_para.toString().c_str());

    check_cuda_value(cudaDeviceSynchronize());

    // deserialize all ranks' cudaIpcMemHandle
    std::vector<cudaIpcMemHandle_t> handles(comm->tp_size);
    for (size_t i = 0; i < handles.size(); ++i) {
        check_cuda_value(cudaMemcpyAsync(handles[i].reserved,
                                         serial_handle_buffer_ptr + CUDA_IPC_HANDLE_SIZE * i,
                                         CUDA_IPC_HANDLE_SIZE,
                                         cudaMemcpyDeviceToHost,
                                         stream));
    }

    for (int i = 0; i < comm->tp_size; i++) {
        if (i != comm->myrank) {
            check_cuda_value(cudaIpcOpenMemHandle(&(comm->peer_ptr[hndl][i]),
                                                  handles[i],  // NOLINT(*)
                                                  cudaIpcMemLazyEnablePeerAccess));
        }
    }

    comm->peer_ptr[hndl][comm->myrank] = *gpubuff;
    check_cuda_value(cudaDeviceSynchronize());
    check_cuda_value(cudaFreeAsync(serial_handle_buffer_ptr, stream));
    comm->mem_ptr[hndl] = *gpubuff;
    return comm->free_region++;
}

std::unique_ptr<CommBuffer> initCommBuffer(std::vector<size_t>&       buffer_shape,
                                           DataType                   buffer_type,
                                           const NcclParam&           nccl_para,
                                           const std::vector<size_t>& tp_ranks,
                                           bool                       is_ag,
                                           cudaStream_t               stream) {
    if (!checkP2PAvailable(tp_ranks, nccl_para.rank_)) {
        RTP_LLM_FAIL("P2P is not available for comm_overlap_type 2");
    }

    auto comm =
        std::make_unique<CommBuffer>(nccl_para, buffer_shape, buffer_type, tp_ranks, nccl_para.rank_, is_ag, stream);
    return comm;
}

void invokeLocalReduceDispatch(
    DataType data_type, void* inputs, void* output, int num_inputs, int input_size, cudaStream_t stream) {
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, localReduce, inputs, output, num_inputs, input_size, stream);
}

}  // namespace rtp_llm
