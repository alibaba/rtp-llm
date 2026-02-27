#pragma once

#include <torch/torch.h>
#include <torch/extension.h>
#include <tuple>
#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace rtp_llm {

/**
 * @brief Wrapper function: Convert void* buffer pointer to int64_t for Python
 *
 * @param size Buffer size in bytes
 * @return Tuple of (buffer_address_as_int64, ipc_handle_tensor)
 */
std::tuple<int64_t, at::Tensor> allocate_shared_buffer_py(int64_t size);

/**
 * @brief Wrapper function: Convert void* IPC pointer to int64_t for Python
 *
 * @param mem_handle IPC memory handle tensor from remote process
 * @return Buffer address as int64_t (GPU device pointer)
 */
int64_t open_ipc_handle_py(at::Tensor& mem_handle);

/**
 * @brief Wrapper function: Initialize communicator with IPC pointers
 *
 * @param ipc_ptrs List of IPC memory pointers from remote processes
 * @param local_rank Local rank of current process
 * @param world_size Total number of processes
 * @return Communicator pointer as int64_t (GPU device address)
 */
int64_t init_communicator_py(int64_t local_rank, int64_t world_size);

/**
 * @brief Wrapper function: Dispose communicator
 *
 * @param comm_ptr Communicator pointer as int64_t
 */
void dispose_communicator_py(int64_t comm_address);

/**
 * @brief Wrapper function: Register buffer to communicator with int64_t conversion
 *
 * @param comm_ptr Communicator pointer as int64_t
 * @param buffer_ptrs List of buffer pointers as int64_t
 * @return Handle for registered buffer region
 */
int register_buffer_to_communicator_py(int64_t comm_ptr, std::vector<int64_t> buffer_ptrs);

/**
 * @brief Wrapper function: Send data via user buffers with pointer conversion
 *
 * @param tensor Tensor to send data from
 * @param handler Handler ID for the registered buffer region
 * @param srcoffset Source offset in buffer
 * @param dstoffset Destination offset in buffer
 * @param bytes Number of bytes to send
 * @param comm_ptr Communicator pointer as int64_t
 * @param peer Peer rank to send to
 * @param stream CUDA stream handle as int64_t
 */
void userbuffers_send_py(at::Tensor& tensor,
                         int64_t     handler,
                         int64_t     srcoffset,
                         int64_t     dstoffset,
                         int64_t     bytes,
                         int64_t     comm_ptr,
                         int64_t     peer,
                         int64_t     stream);

/**
 * @brief Wrapper function: Receive data via user buffers with pointer conversion
 *
 * @param tensor Tensor to receive data into
 * @param handler Handler ID for the registered buffer region
 * @param comm_ptr Communicator pointer as int64_t
 * @param peer Peer rank to receive from
 * @param stream CUDA stream handle as int64_t
 */
void userbuffers_recv_py(at::Tensor& tensor,
                         int64_t     handler,
                         int64_t     src_offset,
                         int64_t     dst_offset,
                         int64_t     comm_ptr,
                         int64_t     peer,
                         int64_t     stream);

/**
 * @brief Wrapper function: Ring all-gather operation via user buffers with pointer conversion
 *
 * @param all_gather_tensor Output tensor to store all-gathered data
 * @param tensor Input tensor to be gathered
 * @param handler Handler ID for the registered buffer region
 * @param rank_offsets List of rank offsets for all ranks
 * @param comm_ptr Communicator pointer as int64_t
 * @param send_streams List of send CUDA stream handles as int64_t
 * @param recv_stream Receive CUDA stream handle as int64_t
 */
at::Tensor userbuffers_ring_all_gather_py(at::Tensor&          all_gather_tensor,
                                          at::Tensor&          tensor,
                                          int64_t              handler,
                                          std::vector<int64_t> rank_offsets,
                                          int64_t              comm_ptr,
                                          std::vector<int64_t> send_streams,
                                          int64_t              recv_stream);

/**
 * @brief Register user buffer operations bindings to Python module
 *
 * @param m PyBind11 module reference
 */
void registerUserBuffersOp(py::module_& m);

}  // namespace rtp_llm
