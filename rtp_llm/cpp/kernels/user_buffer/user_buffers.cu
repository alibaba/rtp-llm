#include "rtp_llm/cpp/kernels/user_buffer/user_buffers.h"

namespace rtp_llm {

namespace user_buffers {

__global__ void kuserbuffers_inc(int* id) {
    atomicAdd_system(id, 1);
}

__global__ void
kuserbuffers_pushrecv(int myrank, int peer, int* recv_id, int* flagptr, int adder, uint64_t ub_timeout) {
    const int signal_id = (*recv_id) + adder;
    *recv_id            = signal_id;
    volatile int* flag  = (volatile int*)flagptr;
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

void userbuffers_send(const int       handler,
                      const size_t    srcoffset,
                      const size_t    dstoffset,
                      const size_t    bytes,
                      UbCommunicator* comm,
                      const int       peer,
                      cudaStream_t    stream) {
    int   peerlocal = peer % comm->world_size;
    void* flagptr   = UB_GET_SEND_PTR_BY_INDEX(peerlocal, comm);
    assert(UB_INTRANODE(peerlocal));
    void* srcptr = reinterpret_cast<char*>(comm->mem_ptr[handler]) + srcoffset;
    void* dstptr = reinterpret_cast<char*>(comm->peer_ptr[handler][peerlocal]) + dstoffset;
    AT_CUDA_CHECK(cudaMemcpyAsync(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice, stream));
    kuserbuffers_inc<<<1, 1, 0, stream>>>(reinterpret_cast<int*>(flagptr));
}

void userbuffers_recv(const int handler, UbCommunicator* comm, const int peer, cudaStream_t stream) {
    int   peerlocal = peer % comm->world_size;
    void* flagptr   = UB_GET_RECV_PTR_BY_INDEX(peer, comm);

    assert(UB_INTRANODE(peerlocal));
    kuserbuffers_pushrecv<<<1, 1, 0, stream>>>(comm->local_rank,
                                               peerlocal,
                                               &comm->recv_id[peer * NVTE_MAX_REGIONS + handler],
                                               reinterpret_cast<int*>(flagptr),
                                               true,
                                               comm->ub_timeout);
}

void* init_communicator(int64_t local_rank, int64_t world_size) {
    UbCommunicator* comm = new UbCommunicator();
    comm->local_rank     = static_cast<int32_t>(local_rank);
    comm->world_size     = static_cast<int32_t>(world_size);

    int device_clock = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(&device_clock, cudaDevAttrClockRate, local_rank));
    comm->ub_timeout  = 1000ull * device_clock * 110;
    comm->free_region = 0;
    comm->gpu_ptrs    = 0;
    AT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&comm->send_id), comm->world_size * sizeof(int)));
    AT_CUDA_CHECK(cudaMemset(comm->send_id, 0, comm->world_size * sizeof(int)));
    AT_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&comm->recv_id), NVTE_MAX_REGIONS * comm->world_size * sizeof(int)));
    AT_CUDA_CHECK(cudaMemset(comm->recv_id, 0, NVTE_MAX_REGIONS * comm->world_size * sizeof(int)));
    return comm;
}

void destory_communicator(UbCommunicator* comm) {

    for (int hndl = 0; hndl < comm->free_region; hndl++) {
        for (int rank = 0; rank < comm->world_size; rank++) {
            if (rank != comm->local_rank) {
                cudaIpcCloseMemHandle(comm->peer_ptr[hndl][rank]);
            } else {
                AT_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(comm->peer_ptr[hndl][rank])));
            }
        }
        free(comm->peer_ptr[hndl]);
    }
    AT_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(comm->recv_id)));
    AT_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(comm->send_id)));
}

int register_buffer_to_communicator(void* comm_ptr, std::vector<void*> buffer_ptrs) {
    UbCommunicator* comm = reinterpret_cast<UbCommunicator*>(comm_ptr);
    int             hndl = comm->free_region;
    comm->peer_ptr[hndl] = reinterpret_cast<void**>(malloc(sizeof(void*) * (comm->world_size)));

    for (int i = 0; i < comm->world_size; i++) {
        comm->peer_ptr[hndl][i] = reinterpret_cast<void*>(buffer_ptrs[i]);
    }
    comm->mem_ptr[hndl] = reinterpret_cast<void*>(buffer_ptrs[comm->local_rank]);
    return comm->free_region++;
}

std::tuple<void*, at::Tensor> allocate_shared_buffer_and_handle(int64_t size) {
    auto            device_index = at::cuda::current_device();
    at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
    void*           buffer;
    auto            stream = at::cuda::getCurrentCUDAStream().stream();

    AT_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
    AT_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto options = at::TensorOptions().dtype(at::kByte).device(at::kCPU);
    auto handle  = at::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
    AT_CUDA_CHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)handle.data_ptr(), buffer));

    return std::make_tuple(buffer, handle);
}

void* open_mem_handle(at::Tensor& mem_handle) {
    void* ipc_ptr;
    AT_CUDA_CHECK(cudaIpcOpenMemHandle(
        (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)mem_handle.data_ptr()), cudaIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<void*>(ipc_ptr);
}

}  // namespace user_buffers
}  // namespace rtp_llm