#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/runtime/DeviceError.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "autil/EnvUtil.h"
#include "autil/StackTracer.h"

#include <atomic>
#include <cstdio>
#include <mutex>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

namespace {
std::atomic<bool>       g_runtime_initialized{false};
std::atomic<bool>       g_enable_comm_overlap{true};
std::atomic<int64_t>    g_device_id{0};
std::atomic<MlaOpsType> g_resolved_mla_ops_type{MlaOpsType::AUTO};
std::mutex              g_runtime_mutex;

MlaOpsType resolveMlaOpsType(MlaOpsType requested) {
#if USING_CUDA
    if (requested == MlaOpsType::AUTO) {
        auto* prop = at::cuda::getCurrentDeviceProperties();
        return prop->major >= 9 ? MlaOpsType::FLASH_MLA : MlaOpsType::FLASH_INFER;
    }
#endif
    return requested;
}
}  // anonymous namespace

bool isRuntimeInitialized() {
    return g_runtime_initialized.load(std::memory_order_acquire);
}

bool getEnableCommOverlap() {
    return g_enable_comm_overlap.load(std::memory_order_relaxed);
}

int64_t getDeviceId() {
    return g_device_id.load(std::memory_order_relaxed);
}

MlaOpsType initRuntime(std::size_t device_id, bool trace_memory, bool enable_comm_overlap, MlaOpsType mla_ops_type) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (g_runtime_initialized.load(std::memory_order_acquire)) {
        const auto resolved_mla_ops_type = g_resolved_mla_ops_type.load(std::memory_order_relaxed);
        RTP_LLM_LOG_WARNING("Runtime already initialized; ignoring requested device_id=%zu, comm_overlap=%d, "
                            "mla_ops_type=%d. Active device_id=%ld, comm_overlap=%d, mla_ops_type=%d.",
                            device_id,
                            enable_comm_overlap,
                            static_cast<int>(mla_ops_type),
                            g_device_id.load(std::memory_order_relaxed),
                            g_enable_comm_overlap.load(std::memory_order_relaxed),
                            static_cast<int>(resolved_mla_ops_type));
        return resolved_mla_ops_type;
    }

    setlinebuf(stdout);
    if (trace_memory) {
        autil::EnvUtil::setEnv("STACK_TRACER_LOG", "true");
        DECLARE_STACK_TRACER_FILE("rtp_llm_stack.log");
    }

#if USING_CUDA
    RTP_LLM_LOG_INFO("Initialize runtime. device_id=%zu", device_id);
    RTP_LLM_DEVICE_CHECK(cudaSetDevice(device_id));
    at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());
#elif USING_ROCM
    RTP_LLM_LOG_INFO("Initialize runtime (ROCm). device_id=%zu", device_id);
    RTP_LLM_DEVICE_CHECK(hipSetDevice(device_id));
#endif

    g_enable_comm_overlap.store(enable_comm_overlap, std::memory_order_relaxed);
    g_device_id.store(static_cast<int64_t>(device_id), std::memory_order_relaxed);
    const auto resolved_mla_ops_type = resolveMlaOpsType(mla_ops_type);
    g_resolved_mla_ops_type.store(resolved_mla_ops_type, std::memory_order_relaxed);
    g_runtime_initialized.store(true, std::memory_order_release);
    RTP_LLM_LOG_INFO("Runtime init done");
    return resolved_mla_ops_type;
}

// ============================================================
// Sync / check
// ============================================================

#if USING_CUDA

void runtimeSyncAndCheck() {
    RTP_LLM_DEVICE_CHECK(cudaDeviceSynchronize());
}

#elif USING_ROCM

void runtimeSyncAndCheck() {
    RTP_LLM_DEVICE_CHECK(hipDeviceSynchronize());
}

#else

void runtimeSyncAndCheck() {}

#endif

void cudaSyncAndCheck() {
    runtimeSyncAndCheck();
}

void cudaCheckLastError() {
#if USING_CUDA
    RTP_LLM_DEVICE_CHECK_DEBUG(at::cuda::getCurrentCUDAStream().stream());
#elif USING_ROCM
    RTP_LLM_DEVICE_CHECK_DEBUG(at::hip::getCurrentHIPStream().stream());
#endif
}

void cudaPreRun(int device_id) {
    setCurrentThreadDevice(device_id);
}

// ============================================================
// Profiling
// ============================================================

void cudaProfilerBegin() {
#if USING_CUDA
    RTP_LLM_DEVICE_CHECK(cudaProfilerStart());
#endif
}

void cudaProfilerEnd() {
#if USING_CUDA
    RTP_LLM_DEVICE_CHECK(cudaProfilerStop());
#endif
}

// ============================================================
// Events
// ============================================================

#if USING_CUDA

std::shared_ptr<torch::Event> runtimeCreateEvent() {
    auto event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(at::cuda::getCurrentCUDAStream());
    return event;
}

#elif USING_ROCM

std::shared_ptr<torch::Event> runtimeCreateEvent() {
    auto event = std::make_shared<torch::Event>(torch::kHIP);
    event->record(at::hip::getCurrentHIPStream(at::hip::current_device()));
    return event;
}

#else

std::shared_ptr<torch::Event> runtimeCreateEvent() {
    RTP_LLM_FAIL("runtimeCreateEvent requires a CUDA or ROCm build");
}

#endif

// ============================================================
// Status queries
// ============================================================

ExecStatus getGpuExecStatus() {
    MemoryStatus mem;
    size_t       total_bytes = 0;
#if USING_CUDA
    auto error = cudaMemGetInfo(&mem.free_bytes, &total_bytes);
    RTP_LLM_CHECK(error == cudaSuccess);
#elif USING_ROCM
    hipMemGetInfo(&mem.free_bytes, &total_bytes);
#else
    RTP_LLM_FAIL("getGpuExecStatus requires a CUDA or ROCm build");
#endif
    mem.used_bytes      = total_bytes - mem.free_bytes;
    mem.available_bytes = mem.free_bytes;
    ExecStatus status;
    status.device_memory_status = mem;
    return status;
}

}  // namespace rtp_llm
