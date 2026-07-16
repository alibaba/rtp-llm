#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/runtime/Bootstrap.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <atomic>
#include <mutex>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

// ============================================================
// Legacy state — read by getDeviceId / getEnableCommOverlap /
// isRuntimeInitialized accessors. Set by initRuntime().
// ============================================================

namespace {
std::atomic<bool> g_runtime_initialized{false};
bool              g_enable_comm_overlap = true;
int64_t           g_device_id           = 0;
std::mutex        g_runtime_mutex;
}  // anonymous namespace

bool isRuntimeInitialized() {
    return g_runtime_initialized.load(std::memory_order_acquire);
}

bool getEnableCommOverlap() {
    return g_enable_comm_overlap;
}

int64_t getDeviceId() {
    return g_device_id;
}

MlaOpsType resolveMlaOpsType(MlaOpsType requested) {
#if USING_CUDA
    if (requested == MlaOpsType::AUTO) {
        auto* prop = at::cuda::getCurrentDeviceProperties();
        return prop->major >= 9 ? MlaOpsType::FLASH_MLA : MlaOpsType::FLASH_INFER;
    }
#endif
    return requested;
}

MlaOpsType initRuntime(std::size_t device_id, bool trace_memory, bool enable_comm_overlap, MlaOpsType mla_ops_type) {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (g_runtime_initialized.load(std::memory_order_acquire)) {
        RTP_LLM_LOG_WARNING("Runtime is already initialized! will do nothing.");
        return mla_ops_type;
    }

    process::bootstrap({device_id, trace_memory});
    g_enable_comm_overlap = enable_comm_overlap;
    g_device_id           = static_cast<int64_t>(device_id);
    g_runtime_initialized.store(true, std::memory_order_release);
    return resolveMlaOpsType(mla_ops_type);
}

// ============================================================
// Sync / check
// ============================================================

#if USING_CUDA

void runtimeSyncAndCheck() {
    check_cuda_value(cudaDeviceSynchronize());
    check_cuda_error();
}

#else  // ROCm

void runtimeSyncAndCheck() {
    ROCM_CHECK(hipDeviceSynchronize());
    ROCM_CHECK_ERROR();
}

#endif  // USING_CUDA

void cudaSyncAndCheck() {
    runtimeSyncAndCheck();
}

void cudaCheckLastError() {
#if USING_CUDA
    check_cuda_error();
#elif USING_ROCM
    auto err = hipGetLastError();
    if (err != hipSuccess) {
        RTP_LLM_LOG_ERROR("ROCm error: %s", hipGetErrorString(err));
    }
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
    check_cuda_value(cudaProfilerStart());
#endif
}

void cudaProfilerEnd() {
#if USING_CUDA
    check_cuda_value(cudaProfilerStop());
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

#else  // ROCm

std::shared_ptr<torch::Event> runtimeCreateEvent() {
    auto event = std::make_shared<torch::Event>(torch::kHIP);
    event->record(at::hip::getCurrentHIPStream(at::hip::current_device()));
    return event;
}

#endif  // USING_CUDA

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
#endif
    mem.used_bytes      = total_bytes - mem.free_bytes;
    mem.available_bytes = mem.free_bytes;
    ExecStatus status;
    status.device_memory_status = mem;
    return status;
}

}  // namespace rtp_llm
