#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <c10/core/InferenceMode.h>
#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#elif USING_ROCM
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <hip/hip_runtime.h>
#endif
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::sleep_memory {
namespace {

class SleepMemoryDeviceGuard {
public:
    explicit SleepMemoryDeviceGuard(int64_t local_rank) {
#if USING_CUDA
        guard_.emplace(static_cast<int>(local_rank));
#elif USING_ROCM
        guard_.emplace(static_cast<int>(local_rank));
#else
        (void)local_rank;
#endif
    }

private:
#if USING_CUDA
    std::optional<at::cuda::CUDAGuard> guard_;
#elif USING_ROCM
    std::optional<c10::hip::HIPGuardMasqueradingAsCUDA> guard_;
#endif
};

}  // namespace

bool synchronizeSleepDevice(const char* stage) {
#if USING_CUDA
    const auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        RTP_LLM_LOG_ERROR("sleep device synchronize failed at %s: %s", stage, cudaGetErrorString(err));
        cudaGetLastError();  // clear the sticky error so it cannot resurface on an unrelated later CUDA call
        return false;
    }
#elif USING_ROCM
    const auto err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        RTP_LLM_LOG_ERROR("sleep device synchronize failed at %s: %s", stage, hipGetErrorString(err));
        hipGetLastError();  // clear the sticky error so it cannot resurface on an unrelated later HIP call
        return false;
    }
#else
    (void)stage;
#endif
    return true;
}

// Return completely free caching-allocator segments to the driver.  This is
// deliberately a best-effort boundary: Graph/TMS live blocks stay allocated,
// while ordinary free segments are released.  The caller records the result
// but never turns a trim failure into a lifecycle failure.
bool trimSleepAllocator(int64_t local_rank, int64_t epoch, int64_t world_rank, bool graph_enabled, const char* phase) {
#if USING_CUDA || USING_ROCM
    const auto started = std::chrono::steady_clock::now();
    bool       ok      = true;
    {
        try {
            SleepMemoryDeviceGuard device_guard(local_rank);
            c10::cuda::CUDACachingAllocator::emptyCache();
        } catch (const std::exception& e) {
            ok = false;
#if USING_CUDA
            (void)cudaGetLastError();
#elif USING_ROCM
            (void)hipGetLastError();
#endif
            RTP_LLM_LOG_WARNING("[SleepMem] phase=%s status=error graph_baked=%d detail=%s",
                                phase,
                                static_cast<int>(graph_enabled),
                                e.what());
        } catch (...) {
            ok = false;
#if USING_CUDA
            (void)cudaGetLastError();
#elif USING_ROCM
            (void)hipGetLastError();
#endif
            RTP_LLM_LOG_WARNING("[SleepMem] phase=%s status=error graph_baked=%d detail=unknown",
                                phase,
                                static_cast<int>(graph_enabled));
        }
    }
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    RTP_LLM_LOG_INFO(
        "[SleepTiming] op=sleep scope=backend phase=%s status=%s elapsed_ms=%.3f total_ms=%.3f epoch=%ld world_rank=%ld graph_baked=%d",
        phase,
        ok ? "ok" : "error",
        elapsed_ms,
        elapsed_ms,
        epoch,
        world_rank,
        static_cast<int>(graph_enabled));
    return ok;
#else
    (void)local_rank;
    (void)epoch;
    (void)world_rank;
    (void)graph_enabled;
    (void)phase;
    return true;
#endif
}

// Parse selected "Key:  <value> kB" lines from a /proc file (/proc/self/status or
// /proc/meminfo). Returns MiB for each requested key; a key that is missing (or the
// file unreadable) stays -1. Best-effort: never throws.
std::unordered_map<std::string, long> readProcMemKb(const char* path, const std::vector<std::string>& keys) {
    std::unordered_map<std::string, long> out;
    for (const auto& k : keys) {
        out[k] = -1;
    }
    std::ifstream fin(path);
    if (!fin.is_open()) {
        return out;
    }
    std::string line;
    while (std::getline(fin, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, colon);
        auto              it  = out.find(key);
        if (it == out.end()) {
            continue;
        }
        // value is like "  12345 kB"; extract the first integer -> MiB.
        long value_kb = 0;
        if (sscanf(line.c_str() + colon + 1, "%ld", &value_kb) == 1) {
            it->second = value_kb / 1024;
        }
    }
    return out;
}

// Log a detailed one-line memory snapshot (GPU + process pinned/RSS + system) tagged by
// `phase`, so a full sleep/wake cycle leaves a grep-able "[SleepMem]" trail in engine.log.
// Runs only on the sleep/wake hook path (already gated by sleep mode). No config toggle.
// - GPU: getGpuExecStatus() (cudaMemGetInfo on the current device; hip on ROCm).
// - process /proc/self/status: VmRSS (resident), VmHWM (peak), VmLck (mlock'd),
//   VmPin (kernel-pinned pages, e.g. RDMA ibv_reg_mr).
// - system /proc/meminfo: MemAvailable (reclaim-aware; the ONLY signal for cudaHostAlloc
//   pinned host RAM such as level-1 weight backup / memory-cache buffer), MemFree,
//   Mlocked, Cached.
void logSleepMemorySnapshotForRank(const std::string& phase, const ParallelismConfig& parallelism, int64_t epoch) {
    size_t gpu_used_mb = 0, gpu_free_mb = 0, gpu_total_mb = 0;
    // torch caching-allocator counters. NOTE: under torch_memory_saver these track the
    // VIRTUAL address reservation, not physical residency -- on VMM pause the physical
    // pages are freed but the VA (and torch's block bookkeeping) stay, so torch_reserved/
    // torch_alloc hold ~constant across a sleep while gpu_used (cudaMemGetInfo, physical)
    // collapses. Do NOT compute a "non-torch floor" as gpu_used-torch_reserved. They are
    // still useful signals: (torch_reserved - torch_alloc) is the reclaimable free-cache
    // headroom (what emptyCache can return), and a drop in torch_alloc means tensors were
    // actually freed (not merely VMM-paused). -1 when the stats are unavailable.
    long torch_reserved_mb = -1, torch_alloc_mb = -1;
    // Best-effort observability only: getGpuExecStatus()/getDeviceStats() can throw
    // c10::Error. This runs INSIDE the sleep/wake release/restore hooks (via
    // the lifecycle wrapper), so a throw here would be mistaken for a hook failure and push
    // the controller to the terminal ERROR state on an otherwise-successful sleep. Swallow
    // it: leave the fields at their sentinels and still log what we have.
    try {
        SleepMemoryDeviceGuard device_guard(parallelism.local_rank);
        const auto             mem = getGpuExecStatus().device_memory_status;
        gpu_used_mb                = mem.used_bytes / 1024 / 1024;
        gpu_free_mb                = mem.free_bytes / 1024 / 1024;
        gpu_total_mb               = (mem.used_bytes + mem.free_bytes) / 1024 / 1024;
#if USING_CUDA || USING_ROCM
        const auto stats  = c10::cuda::CUDACachingAllocator::getDeviceStats(at::cuda::current_device());
        torch_reserved_mb = stats.reserved_bytes[0].current / 1024 / 1024;
        torch_alloc_mb    = stats.allocated_bytes[0].current / 1024 / 1024;
#endif
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("[SleepMem] phase=%s status=error detail=%s", phase.c_str(), e.what());
    }
    const auto proc = readProcMemKb("/proc/self/status", {"VmRSS", "VmHWM", "VmLck", "VmPin"});
    const auto sys  = readProcMemKb("/proc/meminfo", {"MemAvailable", "MemFree", "Mlocked", "Cached"});
    RTP_LLM_LOG_INFO("[SleepMem] phase=%s epoch=%ld world_rank=%ld local_rank=%ld dp_rank=%ld tp_rank=%ld ep_rank=%ld "
                     "gpu_used=%zuMiB gpu_free=%zuMiB gpu_total=%zuMiB "
                     "torch_reserved=%ldMiB torch_alloc=%ldMiB "
                     "| proc VmRSS=%ldMiB VmHWM=%ldMiB VmLck=%ldMiB VmPin=%ldMiB "
                     "| sys MemAvailable=%ldMiB MemFree=%ldMiB Mlocked=%ldMiB Cached=%ldMiB",
                     phase.c_str(),
                     epoch,
                     parallelism.world_rank,
                     parallelism.local_rank,
                     parallelism.dp_rank,
                     parallelism.tp_rank,
                     parallelism.ep_rank,
                     gpu_used_mb,
                     gpu_free_mb,
                     gpu_total_mb,
                     torch_reserved_mb,
                     torch_alloc_mb,
                     proc.at("VmRSS"),
                     proc.at("VmHWM"),
                     proc.at("VmLck"),
                     proc.at("VmPin"),
                     sys.at("MemAvailable"),
                     sys.at("MemFree"),
                     sys.at("Mlocked"),
                     sys.at("Cached"));
}

}  // namespace rtp_llm::sleep_memory
