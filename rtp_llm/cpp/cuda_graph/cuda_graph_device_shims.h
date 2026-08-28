#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_ASCEND
#define GRAPH_DEVICE_TYPE torch::kPrivateUse1
#elif USING_ROCM
#include <ATen/hip/HIPGraph.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>
#define GRAPH_DEVICE_TYPE c10::DeviceType::HIP
#else
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#define GRAPH_DEVICE_TYPE c10::DeviceType::CUDA
#endif

namespace py = pybind11;

namespace rtp_llm {
#if USING_ROCM
namespace rocm {
void  setHipGraphCaptureEnabled(bool enabled);
void* getHipGraphTpNcclComm();
}  // namespace rocm
#endif
namespace cuda_graph {

struct GraphNcclCaptureContext {
    uintptr_t comm_handle{0};
    int       rank{0};
    int       world_size{1};
};

enum class GraphMemcpyKind {
    D2D,
    D2H,
    H2D,
};

#if USING_CUDA
using GraphPoolHandle = c10::cuda::MempoolId_t;
#else
struct GraphPoolHandle {};
#endif

#if USING_ASCEND
using GraphStream = void*;
struct GraphStreamGuard {
    explicit GraphStreamGuard(GraphStream) {}
};
#elif USING_ROCM
using GraphStream      = at::hip::HIPStream;
using GraphStreamGuard = at::hip::HIPStreamGuard;
#else
using GraphStream      = at::cuda::CUDAStream;
using GraphStreamGuard = at::cuda::CUDAStreamGuard;
#endif

inline GraphStream toGraphStream(const torch::Stream& stream) {
#if USING_ASCEND
    (void)stream;
    return nullptr;
#elif USING_ROCM
    return at::hip::HIPStream(stream);
#else
    return at::cuda::CUDAStream(stream);
#endif
}

inline void setDevice(int rank) {
#if USING_ASCEND
    (void)rank;
#elif USING_ROCM
    auto result = hipSetDevice(rank);
    RTP_LLM_CHECK_WITH_INFO(result == hipSuccess, "hipSetDevice(%d) failed: %s", rank, hipGetErrorString(result));
    at::hip::set_device(rank);
#else
    check_cuda_value(cudaSetDevice(rank));
    at::cuda::set_device(rank);
#endif
}

inline void* getGraphCaptureTpNcclComm() {
#if USING_ROCM
    return rocm::getHipGraphTpNcclComm();
#else
    return nullptr;
#endif
}

inline GraphStream graphGetStreamFromPool(bool is_high_priority) {
#if USING_ASCEND
    return nullptr;
#elif USING_ROCM
    return at::hip::getStreamFromPool(is_high_priority);
#else
    return at::cuda::getStreamFromPool(is_high_priority);
#endif
}

inline GraphStream graphGetCurrentStream() {
#if USING_ASCEND
    return nullptr;
#elif USING_ROCM
    return at::hip::getCurrentHIPStream(at::hip::current_device());
#else
    return at::cuda::getCurrentCUDAStream(at::cuda::current_device());
#endif
}

inline void graphSetCurrentStream(GraphStream stream) {
#if USING_ASCEND
    (void)stream;
#elif USING_ROCM
    at::hip::setCurrentHIPStream(stream);
#else
    at::cuda::setCurrentCUDAStream(stream);
#endif
}

inline torch::Event makeGraphEvent() {
#if USING_ASCEND
    return torch::Event(torch::kPrivateUse1);
#else
    return torch::Event(GRAPH_DEVICE_TYPE);
#endif
}

// Event/stream ordering helpers. Ascend's GraphStream is an opaque handle, so
// cross-stream record/block pairs degrade to stream-synchronous semantics.
inline void graphRecordEvent(torch::Event& event, GraphStream stream) {
#if USING_ASCEND
    (void)event;
    (void)stream;
#else
    event.record(stream);
#endif
}

inline void graphBlockEvent(const torch::Event& event, GraphStream stream) {
#if USING_ASCEND
    (void)event;
    (void)stream;
#else
    event.block(stream);
#endif
}

#if USING_ROCM
py::module_& getCollectiveTorchModule();
#endif

void            register_graph_capture_nccl_comm(void* nccl_comm, int world_size, int rank);
void            enter_graph_capture(GraphNcclCaptureContext* ctx);
void            exit_graph_capture(GraphNcclCaptureContext* ctx);
void            finish_capture_session();
void            graphMemcpyAsync(void* dst, const void* src, size_t size, GraphMemcpyKind kind, void* stream);
void            graphDeviceSynchronize();
void            graphMemGetInfo(size_t* free_bytes, size_t* total_bytes);
size_t          graphReservedBytes();
size_t          graphAllocatedBytes();
GraphPoolHandle graphPoolHandle();
#if USING_CUDA || USING_ROCM
void            graphCaptureBegin(at::cuda::CUDAGraph& graph, GraphPoolHandle pool);
#else
void            graphCaptureBegin(void* graph, GraphPoolHandle pool);
#endif

}  // namespace cuda_graph
}  // namespace rtp_llm
