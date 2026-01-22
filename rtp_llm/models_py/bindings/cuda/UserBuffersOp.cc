#include "rtp_llm/models_py/bindings/cuda/UserBuffersOp.h"
#include "rtp_llm/cpp/kernels/user_buffer/user_buffers.h"

namespace rtp_llm {

/**
 * @brief Wrapper function: Convert void* buffer pointer to int64_t (GPU device address)
 */
std::tuple<int64_t, at::Tensor> allocate_shared_buffer_py(int64_t size) {
    auto [buffer_ptr, ipc_handle] = user_buffers::allocate_shared_buffer_and_handle(size);
    // Convert void* to int64_t (GPU device pointer address)
    int64_t buffer_address = reinterpret_cast<int64_t>(buffer_ptr);
    return std::make_tuple(buffer_address, ipc_handle);
}

/**
 * @brief Wrapper function: Convert void* IPC pointer to int64_t for Python
 */
int64_t open_ipc_handle_py(at::Tensor& mem_handle) {
    void* ipc_ptr = user_buffers::open_mem_handle(mem_handle);
    // Convert void* to int64_t
    int64_t ipc_address = reinterpret_cast<int64_t>(ipc_ptr);
    return ipc_address;
}

/**
 * @brief Wrapper function: Convert int64_t IPC pointers to void* and initialize communicator
 */
int64_t init_communicator_py(int64_t local_rank, int64_t world_size) {
    void*                         comm_void = user_buffers::init_communicator(local_rank, world_size);
    user_buffers::UbCommunicator* comm      = reinterpret_cast<user_buffers::UbCommunicator*>(comm_void);
    // Convert void* to int64_t (GPU device pointer address)
    int64_t comm_address = reinterpret_cast<int64_t>(comm);
    return comm_address;
}

void dispose_communicator_py(int64_t comm_address) {
    user_buffers::UbCommunicator* comm = reinterpret_cast<user_buffers::UbCommunicator*>(comm_address);
    destory_communicator(comm);
    delete comm;
}

/**
 * @brief Wrapper function: Convert int64_t pointers to void* and register buffer to communicator
 */
int register_buffer_to_communicator_py(int64_t comm_ptr, std::vector<int64_t> buffer_ptrs) {
    // Convert int64_t to void*
    void* comm_void = reinterpret_cast<void*>(comm_ptr);

    // Convert vector of int64_t to vector of void*
    std::vector<void*> buffer_void_ptrs;
    for (int64_t ptr : buffer_ptrs) {
        buffer_void_ptrs.push_back(reinterpret_cast<void*>(ptr));
    }

    // Call the original function with converted pointers
    return user_buffers::register_buffer_to_communicator(comm_void, buffer_void_ptrs);
}

/**
 * @brief Wrapper function: Convert int64_t pointers to void* and call userbuffers_send
 */
void userbuffers_send_py(at::Tensor& tensor,
                         int64_t     handler,
                         int64_t     srcoffset,
                         int64_t     dstoffset,
                         int64_t     bytes,
                         int64_t     comm_ptr,
                         int64_t     peer,
                         int64_t     stream) {

    user_buffers::UbCommunicator* communicator_ptr = reinterpret_cast<user_buffers::UbCommunicator*>(comm_ptr);
    cudaStream_t                  cuda_stream      = reinterpret_cast<cudaStream_t>(stream);

    auto data_ptr = reinterpret_cast<char*>(tensor.data_ptr());

    auto src_buffer_with_offset = reinterpret_cast<char*>(communicator_ptr->mem_ptr[handler]) + srcoffset;

    // copy tensor to buffer
    AT_CUDA_CHECK(cudaMemcpyAsync(src_buffer_with_offset, data_ptr, bytes, cudaMemcpyDeviceToDevice, cuda_stream));

    // copy tensor to peer buffer with offset
    user_buffers::userbuffers_send(static_cast<int>(handler),
                                   static_cast<size_t>(srcoffset),
                                   static_cast<size_t>(dstoffset),
                                   static_cast<size_t>(bytes),
                                   communicator_ptr,
                                   static_cast<int>(peer),
                                   cuda_stream);
}

/**
 * @brief Wrapper function: Convert int64_t pointers to void* and call userbuffers_recv
 */
void userbuffers_recv_py(at::Tensor& tensor,
                         int64_t     handler,
                         int64_t     src_offset,
                         int64_t     dst_offset,
                         int64_t     comm_ptr,
                         int64_t     peer,
                         int64_t     stream) {

    user_buffers::UbCommunicator* communicator_ptr = reinterpret_cast<user_buffers::UbCommunicator*>(comm_ptr);
    cudaStream_t                  cuda_stream      = reinterpret_cast<cudaStream_t>(stream);

    // wait for peer to receive data
    user_buffers::userbuffers_recv(static_cast<int>(handler), communicator_ptr, static_cast<int>(peer), cuda_stream);

    auto   dst_buffer    = reinterpret_cast<char*>(communicator_ptr->mem_ptr[handler]);
    auto   data_ptr      = reinterpret_cast<char*>(tensor.data_ptr());
    size_t respect_bytes = tensor.numel() * tensor.element_size();
    // copy buffer to tensor
    AT_CUDA_CHECK(
        cudaMemcpyAsync(data_ptr, dst_buffer + dst_offset, respect_bytes, cudaMemcpyDeviceToDevice, cuda_stream));
}

/**
 * @brief Wrapper function: Ring all-gather operation via user buffers with pointer conversion
 *
 * @param all_gather_tensor Output tensor to store all-gathered data
 * @param tensor Input tensor to be gathered
 * @param handler Handler ID for the registered buffer region
 * @param comm_ptr Communicator pointer as int64_t
 * @param rank_offsets List of rank offsets for all ranks
 * @param send_streams List of send CUDA stream handles as int64_t
 * @param recv_stream Receive CUDA stream handle as int64_t
 */
at::Tensor userbuffers_ring_all_gather_py(at::Tensor&          all_gather_tensor,
                                          at::Tensor&          tensor,
                                          int64_t              handler,
                                          std::vector<int64_t> rank_offsets,
                                          int64_t              comm_ptr,
                                          std::vector<int64_t> send_streams,
                                          int64_t              recv_stream) {
    user_buffers::UbCommunicator* communicator_ptr = reinterpret_cast<user_buffers::UbCommunicator*>(comm_ptr);
    cudaStream_t                  recv_cuda_stream = reinterpret_cast<cudaStream_t>(recv_stream);
    int                           world_size       = communicator_ptr->world_size;
    int                           local_rank       = communicator_ptr->local_rank;

    // Convert send stream IDs to cudaStream_t
    std::vector<cudaStream_t> send_cuda_streams;
    for (int64_t stream_id : send_streams) {
        send_cuda_streams.push_back(reinterpret_cast<cudaStream_t>(stream_id));
    }

    int    buffer_handle = static_cast<int>(handler);
    size_t bytes         = tensor.numel() * tensor.element_size();
    auto   tensor_ptr    = reinterpret_cast<char*>(tensor.data_ptr());
    auto   src_buffer    = reinterpret_cast<char*>(communicator_ptr->mem_ptr[buffer_handle]);

    cudaStream_t current_stream         = at::cuda::getCurrentCUDAStream().stream();
    auto         src_buffer_with_offset = src_buffer + rank_offsets[local_rank];
    AT_CUDA_CHECK(cudaMemcpyAsync(src_buffer_with_offset, tensor_ptr, bytes, cudaMemcpyDeviceToDevice, current_stream));

    // Create event to synchronize current_stream with send_streams
    cudaEvent_t memcpy_done_event;
    AT_CUDA_CHECK(cudaEventCreate(&memcpy_done_event));
    AT_CUDA_CHECK(cudaEventRecord(memcpy_done_event, current_stream));

    auto output_data_ptr = reinterpret_cast<char*>(all_gather_tensor.data_ptr());
    // memcpy local partition
    AT_CUDA_CHECK(cudaMemcpyAsync(
        output_data_ptr + bytes * local_rank, src_buffer_with_offset, bytes, cudaMemcpyDeviceToDevice, current_stream));
    AT_CUDA_CHECK(cudaStreamWaitEvent(recv_cuda_stream, memcpy_done_event, 0));

    for (int i = 0; i < world_size - 1; i++) {
        int next_rank = (local_rank + i + 1) % world_size;
        int prev_rank = (local_rank - i - 1 + world_size) % world_size;
        // Make send_stream wait for memcpy to complete
        AT_CUDA_CHECK(cudaStreamWaitEvent(send_cuda_streams[next_rank], memcpy_done_event, 0));
        user_buffers::userbuffers_send(buffer_handle,
                                       static_cast<size_t>(rank_offsets[local_rank]),
                                       static_cast<size_t>(rank_offsets[local_rank]),
                                       static_cast<size_t>(bytes),
                                       communicator_ptr,
                                       static_cast<int>(next_rank),
                                       send_cuda_streams[next_rank]);

        auto dst_address = src_buffer + rank_offsets[prev_rank];
        user_buffers::userbuffers_recv(buffer_handle, communicator_ptr, static_cast<int>(prev_rank), recv_cuda_stream);
        AT_CUDA_CHECK(cudaMemcpyAsync(
            output_data_ptr + bytes * prev_rank, dst_address, bytes, cudaMemcpyDeviceToDevice, recv_cuda_stream));
    }
    // Clean up event
    AT_CUDA_CHECK(cudaEventDestroy(memcpy_done_event));

    return all_gather_tensor;
}

void registerUserBuffersOp(py::module_& m) {
    // Allocate shared buffer with IPC handle
    // Returns: Tuple of (buffer_address: int64_t, ipc_handle: torch.Tensor)
    m.def("allocate_shared_buffer",
          &allocate_shared_buffer_py,
          "Allocate shared CUDA buffer with IPC handle for inter-process communication",
          py::arg("size"));

    m.def("open_ipc_handle",
          &open_ipc_handle_py,
          "Open IPC memory handle to access shared buffer from another process",
          py::arg("mem_handle"));

    // Initialize communicator with IPC pointers
    // Args: ipc_ptrs (List[int64_t]), local_rank (int64_t), world_size (int64_t)
    // Returns: communicator_address (int64_t)
    m.def("init_communicator",
          &init_communicator_py,
          "Initialize UbCommunicator with IPC pointers from remote processes",
          py::arg("local_rank"),
          py::arg("world_size"));

    // Dispose communicator with address
    // Args: comm_ptr (int64_t)
    m.def("dispose_communicator",
          &dispose_communicator_py,
          "Dispose UbCommunicator with python address and release resources",
          py::arg("comm_ptr"));

    // Register buffer to communicator
    // Args: comm_ptr (int64_t), buffer_ptrs (List[int64_t])
    // Returns: handle (int) for registered buffer region
    m.def("register_buffer_to_communicator",
          &register_buffer_to_communicator_py,
          "Register buffers to communicator for inter-process communication",
          py::arg("comm_ptr"),
          py::arg("buffer_ptrs"));

    // Send data via user buffers
    // Args: tensor (at::tensor), handler (int), srcoffset (size_t), dstoffset (size_t), bytes (size_t),
    //       comm_ptr (int64_t), peer (int), stream (int64_t)
    m.def("userbuffers_send",
          &userbuffers_send_py,
          "Send data via user buffers to peer",
          py::arg("tensor"),
          py::arg("handler"),
          py::arg("srcoffset"),
          py::arg("dstoffset"),
          py::arg("bytes"),
          py::arg("comm_ptr"),
          py::arg("peer"),
          py::arg("stream"));

    // Receive data via user buffers
    // Args: tensor (at::tensor), handler (int), comm_ptr (int64_t), peer (int), stream (int64_t)
    m.def("userbuffers_recv",
          &userbuffers_recv_py,
          "Receive data via user buffers from peer",
          py::arg("tensor"),
          py::arg("handler"),
          py::arg("srcoffset"),
          py::arg("dstoffset"),
          py::arg("comm_ptr"),
          py::arg("peer"),
          py::arg("stream"));

    // Ring all-gather operation via user buffers
    // Args: all_gather_tensor (at::tensor), tensor (at::tensor), handler (int), rank_offsets (List[int64_t]), comm_ptr
    // (int64_t), send_stream_ids (List[int64_t]), recv_stream (int64_t)
    m.def("userbuffers_ring_all_gather",
          &userbuffers_ring_all_gather_py,
          "Ring all-gather operation via user buffers",
          py::arg("all_gather_tensor"),
          py::arg("tensor"),
          py::arg("handler"),
          py::arg("rank_offsets"),
          py::arg("comm_ptr"),
          py::arg("send_stream_ids"),
          py::arg("recv_stream"));
}

}  // namespace rtp_llm
