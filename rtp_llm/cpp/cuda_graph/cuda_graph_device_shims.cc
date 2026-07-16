#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#if USING_ROCM
#include <hip/hip_runtime.h>
#else
#include <c10/cuda/CUDACachingAllocator.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#endif

namespace rtp_llm {
namespace cuda_graph {

namespace {
#if USING_ROCM
inline void graphCheck(hipError_t result, const char* call_expr) {
    RTP_LLM_CHECK_WITH_INFO(result == hipSuccess, "%s failed: %s", call_expr, hipGetErrorString(result));
}
#else
inline void graphCheck(cudaError_t result, const char* call_expr) {
    (void)call_expr;
    check_cuda_value(result);
}
#endif
}  // namespace

#if USING_ROCM
py::module_& getGraphCaptureModule() {
    RTP_LLM_CHECK_WITH_INFO(PyGILState_Check(), "getGraphCaptureModule requires GIL to be held");
    static py::module_ graph_capture_module = py::module_::import("rtp_llm.models_py.distributed.rocm_rccl");
    return graph_capture_module;
}
#endif

void register_graph_capture_nccl_comm(void* nccl_comm, int world_size, int rank) {
#if USING_ROCM
    py::gil_scoped_acquire gil;
    if (world_size <= 1) {
        try {
            py::module_& graph_capture = getGraphCaptureModule();
            graph_capture.attr("set_graph_capture_nccl_comm")(static_cast<uintptr_t>(0), 0, rank);
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_WARNING("Failed to clear NCCL comm for graph capture: %s", e.what());
        }
        return;
    }
    if (nccl_comm == nullptr) {
        RTP_LLM_LOG_INFO(
            "Skip graph-capture NCCL comm registration for rank=%d: using pre-bootstrapped RCCL communicator", rank);
        return;
    }
    try {
        py::module_& graph_capture = getGraphCaptureModule();
        graph_capture.attr("set_graph_capture_nccl_comm")(reinterpret_cast<uintptr_t>(nccl_comm), world_size, rank);
        RTP_LLM_LOG_INFO("Registered NCCL comm for graph capture (rank=%d, world_size=%d)", rank, world_size);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING(
            "Failed to register NCCL comm for graph capture (rank=%d, world_size=%d): %s", rank, world_size, e.what());
        throw std::runtime_error(std::string("Failed to register NCCL comm for graph capture: ") + e.what());
    }
#else
    (void)nccl_comm;
    (void)world_size;
    (void)rank;
#endif
}

void enter_graph_capture(GraphNcclCaptureContext* ctx) {
#if USING_ROCM
    // State ownership: C++ owns in_hip_graph_capture (atomic bool); Python owns
    // _rccl_comm/_rccl_world_size.  On failure we roll back both sides:
    //   1. setHipGraphCaptureEnabled(false)       -- resets C++ capture flag
    //   2. set_graph_capture_nccl_comm(0, 0, rank) -- triggers Python _clear_hipgraph_capture_nccl_comm()
    py::gil_scoped_acquire gil;
    rocm::setHipGraphCaptureEnabled(true);
    try {
        py::module_& graph_capture = getGraphCaptureModule();
        if (ctx && ctx->comm_handle != 0) {
            graph_capture.attr("enter_graph_capture_mode")(ctx->comm_handle, ctx->world_size, ctx->rank);
        } else {
            graph_capture.attr("enter_graph_capture_mode")(0, 0, 0);
        }
    } catch (const py::error_already_set& e) {
        rocm::setHipGraphCaptureEnabled(false);
        const int rank = ctx ? ctx->rank : -1;
        try {
            py::module_& graph_capture = getGraphCaptureModule();
            graph_capture.attr("set_graph_capture_nccl_comm")(static_cast<uintptr_t>(0), 0, rank);
        } catch (const py::error_already_set& clear_e) {
            RTP_LLM_LOG_WARNING("Failed to clear NCCL comm after enter_graph_capture failure: %s", clear_e.what());
        }
        RTP_LLM_LOG_WARNING("Failed to enter graph capture mode: %s", e.what());
        throw;
    }
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = true;
#endif
}

void exit_graph_capture(GraphNcclCaptureContext* ctx) {
#if USING_ROCM
    // exit_graph_capture_mode() is intentionally a no-op on Python side:
    // _rccl_comm is preserved across capture sessions for reuse in replay.
    // C++ unconditionally calls setHipGraphCaptureEnabled(false) after this
    // function, regardless of success or failure.
    py::gil_scoped_acquire gil;
    try {
        py::module_& graph_capture = getGraphCaptureModule();
        graph_capture.attr("exit_graph_capture_mode")();
    } catch (const py::error_already_set& e) {
        const unsigned long long comm_handle = ctx ? static_cast<unsigned long long>(ctx->comm_handle) : 0ULL;
        const int                rank        = ctx ? ctx->rank : -1;
        const int                world_size  = ctx ? ctx->world_size : -1;
        RTP_LLM_LOG_WARNING("Failed to exit graph capture mode (comm_handle=%llu, rank=%d, world_size=%d): %s",
                            comm_handle,
                            rank,
                            world_size,
                            e.what());
        try {
            py::module_& graph_capture = getGraphCaptureModule();
            graph_capture.attr("set_graph_capture_nccl_comm")(static_cast<uintptr_t>(0), 0, rank);
        } catch (const py::error_already_set& clear_e) {
            RTP_LLM_LOG_WARNING("Failed to clear NCCL comm after exit_graph_capture failure: %s", clear_e.what());
        }
        rocm::setHipGraphCaptureEnabled(false);
        throw;
    }
    rocm::setHipGraphCaptureEnabled(false);
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = false;
#endif
}

void graphMemcpyAsync(void* dst, const void* src, size_t size, GraphMemcpyKind kind, void* stream) {
#if USING_ROCM
    hipMemcpyKind hip_kind = hipMemcpyDeviceToDevice;
    if (kind == GraphMemcpyKind::D2H) {
        hip_kind = hipMemcpyDeviceToHost;
    } else if (kind == GraphMemcpyKind::H2D) {
        hip_kind = hipMemcpyHostToDevice;
    }
    graphCheck(hipMemcpyAsync(dst, src, size, hip_kind, static_cast<hipStream_t>(stream)), "hipMemcpyAsync");
#else
    cudaMemcpyKind cuda_kind = cudaMemcpyDeviceToDevice;
    if (kind == GraphMemcpyKind::D2H) {
        cuda_kind = cudaMemcpyDeviceToHost;
    } else if (kind == GraphMemcpyKind::H2D) {
        cuda_kind = cudaMemcpyHostToDevice;
    }
    graphCheck(cudaMemcpyAsync(dst, src, size, cuda_kind, static_cast<cudaStream_t>(stream)), "cudaMemcpyAsync");
#endif
}

void graphDeviceSynchronize() {
#if USING_ROCM
    graphCheck(hipDeviceSynchronize(), "hipDeviceSynchronize");
#else
    graphCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
#endif
}

void graphMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
#if USING_ROCM
    graphCheck(hipMemGetInfo(free_bytes, total_bytes), "hipMemGetInfo");
#else
    graphCheck(cudaMemGetInfo(free_bytes, total_bytes), "cudaMemGetInfo");
#endif
}

size_t graphReservedBytes() {
#if USING_CUDA
    return c10::cuda::CUDACachingAllocator::getDeviceStats(at::cuda::current_device()).reserved_bytes[0].current;
#else
    return 0;
#endif
}

size_t graphAllocatedBytes() {
#if USING_CUDA
    return c10::cuda::CUDACachingAllocator::getDeviceStats(at::cuda::current_device()).allocated_bytes[0].current;
#else
    return 0;
#endif
}

GraphPoolHandle graphPoolHandle() {
#if USING_CUDA
    return at::cuda::graph_pool_handle();
#else
    return GraphPoolHandle{};
#endif
}

void graphCaptureBegin(at::cuda::CUDAGraph& graph, GraphPoolHandle pool) {
#if USING_CUDA
    graph.capture_begin(pool);
#else
    (void)pool;
    graph.capture_begin();
#endif
}

}  // namespace cuda_graph
}  // namespace rtp_llm
