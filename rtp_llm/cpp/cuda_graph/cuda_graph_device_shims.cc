#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include <cinttypes>
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

GraphLifecycleContext acquire_graph_owner(uintptr_t owner_id) {
    GraphLifecycleContext ctx;
#if USING_ROCM
    py::gil_scoped_acquire gil;
    try {
        py::tuple result = getGraphCaptureModule().attr("acquire_graph_owner")(owner_id).cast<py::tuple>();
        RTP_LLM_CHECK_WITH_INFO(py::len(result) == 2, "acquire_graph_owner must return a 2-tuple");
        ctx.owner_token = result[0].cast<uint64_t>();
        ctx.generation  = result[1].cast<uint64_t>();
    } catch (...) {
        std::exception_ptr original_error = std::current_exception();
        try {
            getGraphCaptureModule().attr("release_graph_owner_after_acquire_failure")(owner_id);
        } catch (const std::exception& rollback_error) {
            RTP_LLM_LOG_ERROR("Failed to rollback undecodable ROCm graph owner for owner_id=%" PRIuPTR ": %s",
                              owner_id,
                              rollback_error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("Failed to rollback undecodable ROCm graph owner for owner_id=%" PRIuPTR, owner_id);
        }
        std::rethrow_exception(original_error);
    }
    if (ctx.owner_token == 0) {
        RTP_LLM_LOG_INFO("ROCm graph owner lease is inactive for owner_id=%" PRIuPTR, owner_id);
    } else {
        RTP_LLM_LOG_INFO(
            "Acquired ROCm graph owner lease token=%" PRIu64 " generation=%" PRIu64, ctx.owner_token, ctx.generation);
    }
#else
    (void)owner_id;
#endif
    return ctx;
}

void begin_capture_planning(const GraphLifecycleContext& ctx) {
#if USING_ROCM
    if (ctx.owner_token == 0) {
        return;
    }
    py::gil_scoped_acquire gil;
    getGraphCaptureModule().attr("begin_capture_planning")(ctx.owner_token, ctx.generation);
#else
    (void)ctx;
#endif
}

void cancel_capture_planning(const GraphLifecycleContext& ctx) {
#if USING_ROCM
    if (ctx.owner_token == 0) {
        return;
    }
    py::gil_scoped_acquire gil;
    getGraphCaptureModule().attr("cancel_capture_planning")(ctx.owner_token, ctx.generation);
#else
    (void)ctx;
#endif
}

void prepare_capture_arena(const GraphLifecycleContext& ctx) {
#if USING_ROCM
    if (ctx.owner_token == 0) {
        return;
    }
    py::gil_scoped_acquire gil;
    getGraphCaptureModule().attr("prepare_capture_arena")(ctx.owner_token, ctx.generation);
#else
    (void)ctx;
#endif
}

void release_graph_owner(const GraphLifecycleContext& ctx) {
#if USING_ROCM
    if (ctx.owner_token == 0) {
        return;
    }
    py::gil_scoped_acquire gil;
    getGraphCaptureModule().attr("release_graph_owner")(ctx.owner_token, ctx.generation);
#else
    (void)ctx;
#endif
}

void enter_graph_capture(const GraphLifecycleContext* ctx) {
#if USING_ROCM
    RTP_LLM_CHECK_WITH_INFO(ctx != nullptr, "ROCm graph capture requires a lifecycle context");
    if (ctx->owner_token == 0) {
        // Degenerate TP has no graph communicator to borrow, but HIPGraph
        // capture itself remains valid and collectives stay on their normal
        // eager implementation.
        rocm::setHipGraphCaptureEnabled(true);
        return;
    }
    py::gil_scoped_acquire gil;
    // Publish the capture flag only after the token/generation is validated.
    getGraphCaptureModule().attr("enter_graph_capture_mode")(ctx->owner_token, ctx->generation);
    rocm::setHipGraphCaptureEnabled(true);
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = true;
#endif
}

void exit_graph_capture(const GraphLifecycleContext* ctx) {
#if USING_ROCM
    // The device flag must never remain set after capture unwinds, even if
    // the context or Python-side state validation reports an error.
    rocm::setHipGraphCaptureEnabled(false);
    RTP_LLM_CHECK_WITH_INFO(ctx != nullptr, "ROCm graph capture requires a lifecycle context");
    if (ctx->owner_token == 0) {
        return;
    }
    py::gil_scoped_acquire gil;
    getGraphCaptureModule().attr("exit_graph_capture_mode")(ctx->owner_token, ctx->generation);
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

void finish_capture_session(const GraphLifecycleContext& ctx) {
#if USING_ROCM
    if (ctx.owner_token == 0) {
        return;
    }
    py::gil_scoped_acquire gil;
    try {
        py::module_& graph_capture = getGraphCaptureModule();
        graph_capture.attr("finish_hipgraph_capture_session")(ctx.owner_token, ctx.generation);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to finish capture session: %s", e.what());
        throw;
    }
#endif
}

}  // namespace cuda_graph
}  // namespace rtp_llm
