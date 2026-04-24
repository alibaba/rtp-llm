#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_ROCM
#include <ATen/hip/HIPGraph.h>
#include <ATen/hip/HIPContext.h>
#define GRAPH_DEVICE_TYPE c10::DeviceType::HIP
#else
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
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

#if USING_ROCM
using GraphStream = at::hip::HIPStream;
#else
using GraphStream = at::cuda::CUDAStream;
#endif

inline void* getGraphCaptureTpNcclComm() {
#if USING_ROCM
    return rocm::getHipGraphTpNcclComm();
#else
    return nullptr;
#endif
}

inline GraphStream graphGetStreamFromPool(bool is_high_priority) {
#if USING_ROCM
    return at::hip::getStreamFromPool(is_high_priority);
#else
    return at::cuda::getStreamFromPool(is_high_priority);
#endif
}

inline GraphStream graphGetCurrentStream() {
#if USING_ROCM
    return at::hip::getCurrentHIPStream(at::hip::current_device());
#else
    return at::cuda::getCurrentCUDAStream(at::cuda::current_device());
#endif
}

inline void graphSetCurrentStream(GraphStream stream) {
#if USING_ROCM
    at::hip::setCurrentHIPStream(stream);
#else
    at::cuda::setCurrentCUDAStream(stream);
#endif
}

inline torch::Event makeGraphEvent() {
    return torch::Event(GRAPH_DEVICE_TYPE);
}

#if USING_ROCM
py::module_& getCollectiveTorchModule();
#endif

void            register_graph_capture_nccl_comm(void* nccl_comm, int world_size, int rank);
void            enter_graph_capture(GraphNcclCaptureContext* ctx);
void            exit_graph_capture(GraphNcclCaptureContext* ctx);
void            graphMemcpyAsync(void* dst, const void* src, size_t size, GraphMemcpyKind kind, void* stream);
void            graphDeviceSynchronize();
void            graphMemGetInfo(size_t* free_bytes, size_t* total_bytes);
size_t          graphReservedBytes();
size_t          graphAllocatedBytes();
GraphPoolHandle graphPoolHandle();
void            graphCaptureBegin(at::cuda::CUDAGraph& graph, GraphPoolHandle pool);

}  // namespace cuda_graph
}  // namespace rtp_llm
