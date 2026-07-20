#include <algorithm>
#include <memory>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <thread>
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
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

using namespace std;

namespace rtp_llm {

namespace {

// Best-effort instance identity for the M4 admission error body: scheduler
// role when deployed (hippo), hostname otherwise.
std::string resolveInstanceId() {
    std::string instance_id = autil::EnvUtil::getEnv("HIPPO_ROLE", "");
    if (instance_id.empty()) {
        char hostname[256] = {0};
        if (gethostname(hostname, sizeof(hostname) - 1) == 0) {
            instance_id = hostname;
        }
    }
    return instance_id;
}

class OptionalSleepDeviceGuard {
public:
    explicit OptionalSleepDeviceGuard(int64_t local_rank) {
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

bool synchronizeSleepDevice(const char* stage) {
#if USING_CUDA
    const auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        RTP_LLM_LOG_ERROR("sleep device synchronize failed at %s: %s", stage, cudaGetErrorString(err));
        return false;
    }
#elif USING_ROCM
    const auto err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        RTP_LLM_LOG_ERROR("sleep device synchronize failed at %s: %s", stage, hipGetErrorString(err));
        return false;
    }
#else
    (void)stage;
#endif
    return true;
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
void logSleepMemorySnapshot(const std::string& phase, int local_rank) {
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
    // invokeHookNoThrow), so a throw here would be mistaken for a hook failure and push
    // the controller to the terminal ERROR state on an otherwise-successful sleep. Swallow
    // it: leave the fields at their sentinels and still log what we have.
    try {
        OptionalSleepDeviceGuard device_guard(local_rank);
        const auto               mem = getGpuExecStatus().device_memory_status;
        gpu_used_mb                  = mem.used_bytes / 1024 / 1024;
        gpu_free_mb                  = mem.free_bytes / 1024 / 1024;
        gpu_total_mb                 = (mem.used_bytes + mem.free_bytes) / 1024 / 1024;
#if USING_CUDA || USING_ROCM
        const auto stats  = c10::cuda::CUDACachingAllocator::getDeviceStats(at::cuda::current_device());
        torch_reserved_mb = stats.reserved_bytes[0].current / 1024 / 1024;
        torch_alloc_mb    = stats.allocated_bytes[0].current / 1024 / 1024;
#endif
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("[SleepMem][%s] GPU stat snapshot failed (ignored): %s", phase.c_str(), e.what());
    }
    const auto proc = readProcMemKb("/proc/self/status", {"VmRSS", "VmHWM", "VmLck", "VmPin"});
    const auto sys  = readProcMemKb("/proc/meminfo", {"MemAvailable", "MemFree", "Mlocked", "Cached"});
    RTP_LLM_LOG_INFO("[SleepMem][%s] rank=%d gpu_used=%zuMiB gpu_free=%zuMiB gpu_total=%zuMiB "
                     "torch_reserved=%ldMiB torch_alloc=%ldMiB "
                     "| proc VmRSS=%ldMiB VmHWM=%ldMiB VmLck=%ldMiB VmPin=%ldMiB "
                     "| sys MemAvailable=%ldMiB MemFree=%ldMiB Mlocked=%ldMiB Cached=%ldMiB",
                     phase.c_str(),
                     local_rank,
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

grpc::Status sleepResultToGrpcStatus(const SleepResult& result) {
    if (result.ok) {
        return grpc::Status::OK;
    }
    switch (result.code) {
        case SleepResult::Code::DISABLED:
        case SleepResult::Code::UNIMPLEMENTED:
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, result.message);
        case SleepResult::Code::INVALID_ARGUMENT:
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, result.message);
        case SleepResult::Code::FAILED_PRECONDITION:
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, result.message);
        case SleepResult::Code::OK:
            return grpc::Status::OK;
    }
    return grpc::Status(grpc::StatusCode::UNKNOWN, result.message);
}

void fillSleepStatusPb(const SleepStatus& status, SleepStatusResponsePB* response) {
    response->set_state(sleepStateToString(status.state));
    response->set_sleep_epoch(status.sleep_epoch);
    response->set_kv_memory_state(status.kv_memory_state);
    response->set_device_kv_cache_valid(status.device_kv_cache_valid);
    response->set_active_request_count(status.active_request_count);
    response->set_active_cache_transfer_count(status.active_cache_transfer_count);
    response->set_gpu_resource_state(status.gpu_resource_state);
    response->set_last_error(status.last_error);
    response->set_sleep_mode_enabled(status.sleep_mode_enabled);
    response->set_effective(status.effective);
    response->set_disabled_reason(status.disabled_reason);
    for (const auto level : status.supported_levels) {
        response->add_supported_levels(level);
    }
    for (const auto& mode : status.supported_modes) {
        response->add_supported_modes(mode);
    }
}

}  // namespace

grpc::Status LocalRpcServer::init(const EngineInitParams&                       maga_init_params,
                                  std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                                  py::object                                    mm_process_engine) {
    meta_.reset(new RpcServerRuntimeMeta());
    maga_init_params_ = maga_init_params;
    weight_manager_   = maga_init_params.weight_manager;
    metrics_reporter_ = maga_init_params.metrics_reporter;
    RTP_LLM_LOG_INFO("LocalRpcServer aux_string %s", maga_init_params_.misc_config.aux_string.c_str());
    propose_maga_init_params_ = propose_params.get();
    if (maga_init_params_.parallelism_config.tp_rank == 0
        && !maga_init_params_.runtime_config.worker_grpc_addrs.empty()) {
        profile_broadcaster_ = std::make_shared<BroadcastManager>(maga_init_params_.runtime_config.worker_grpc_addrs);
        if (!profile_broadcaster_->init()) {
            RTP_LLM_LOG_WARNING("failed to init profile broadcaster");
            profile_broadcaster_.reset();
        }
    }

    {
        pybind11::gil_scoped_release release;
        RTP_LLM_CHECK_WITH_INFO(!PyGILState_Check(),
                                "running engine init with gil held may cause program hang, please check");
        engine_.reset(new NormalEngine(maga_init_params, std::move(propose_params)));
    }
    admission_gate_ = std::make_shared<AdmissionGate>(&engine_->sleepController(), resolveInstanceId());
    installSleepHooks();
    if (maga_init_params.model_config_.mm_model_config.is_multimodal) {
        if (mm_process_engine.is_none()) {
            mm_processor_.reset(new RemoteMultimodalProcessor(maga_init_params.model_config_.mm_model_config,
                                                              maga_init_params.model_config_.max_seq_len,
                                                              metrics_reporter_));
        } else {
            mm_processor_.reset(new LocalMultimodalProcessor(mm_process_engine,
                                                             maga_init_params.model_config_.mm_model_config,
                                                             maga_init_params.model_config_.max_seq_len));
        }
    }

    return grpc::Status::OK;
}

void LocalRpcServer::installSleepHooks() {
    // --- M3: drain counters. ---
    drain_manager_ = std::make_shared<DrainManager>();
    drain_manager_->registerCounter(
        "admission_leases",
        [this]() { return static_cast<size_t>(engine_->sleepController().activeAdmissionCount()); },
        DrainManager::CounterKind::REQUEST);
    auto engine = engine_;
    drain_manager_->registerCounter(
        "scheduler_onflight",
        [engine]() { return static_cast<size_t>(std::max<int64_t>(0, engine->getScheduler().onflightStreams())); },
        DrainManager::CounterKind::REQUEST);
    drain_manager_->registerCounter(
        "rpc_cache_transfer",
        [this]() { return activeCacheTransferCount(); },
        DrainManager::CounterKind::CACHE_TRANSFER);
    if (auto cache_manager = engine_->getCacheManager()) {
        if (auto coordinator = cache_manager->connectorCoordinator()) {
            drain_manager_->registerCounter(
                "connector_inflight",
                [coordinator]() { return static_cast<size_t>(coordinator->inflightTransferCount()); },
                DrainManager::CounterKind::CACHE_TRANSFER);
        }
    }
    drain_manager_->setCancelCallback([this]() {
        const auto cancelled = cancelAbortableStreams();
        RTP_LLM_LOG_INFO("sleep abort callback cancelled %zu non-streaming active stream(s)", cancelled);
    });

    vmm_backend_                   = std::make_shared<VmmBackend>();
    const auto& parallelism_config = maga_init_params_.parallelism_config;
    const bool  runtime_supported  = vmm_backend_->isAvailable();
    std::string disabled_reason;
    if (!vmm_backend_->isAvailable()) {
        disabled_reason =
            "VMM backend is unavailable; start with torch_memory_saver LD_PRELOAD and ENABLE_SLEEP_MODE=1 to enable "
            "sleep mode";
    }
    RTP_LLM_LOG_INFO("sleep hooks: VMM backend available=%d, dp_size=%ld, ep_size=%ld",
                     static_cast<int>(vmm_backend_->isAvailable()),
                     parallelism_config.dp_size,
                     parallelism_config.ep_size);
    engine_->sleepController().setRuntimeSupport(runtime_supported, disabled_reason);

    SleepHooks hooks;
    drain_manager_->installHooks(hooks);  // drain + activeRequestCount + activeCacheTransferCount
    const auto local_rank  = maga_init_params_.parallelism_config.local_rank;
    auto       vmm_backend = vmm_backend_;
    hooks.quiesceEngine    = [engine, local_rank](const SleepOptions& opt) {
        OptionalSleepDeviceGuard device_guard(local_rank);
        // Stall the engine at a TP/DP/EP-safe point before any rank drops GPU memory.
        auto pause_status = engine->pauseAndWaitQuiesced(opt.timeout_ms);
        if (!pause_status.ok()) {
            RTP_LLM_LOG_ERROR("pauseAndWaitQuiesced failed before sleep: %s", pause_status.ToString().c_str());
            return false;
        }
        return true;
    };
    hooks.synchronizeAndDeregisterMr = [this, engine, local_rank](const SleepOptions&) {
        OptionalSleepDeviceGuard device_guard(local_rank);
        // Baseline before any resource is dropped: MR-pinned (VmPin) + KV + weights all live.
        logSleepMemorySnapshot("sleep/RUNNING", local_rank);
        if (!synchronizeSleepDevice("before_dereg_mr")) {
            return false;
        }
        if (auto cache_manager = engine->getCacheManager()) {
            cache_manager->deregUserMr();
        }
        if (!synchronizeSleepDevice("after_dereg_mr")) {
            return false;
        }
        // MR deregistered: VmPin should have collapsed relative to the baseline above.
        logSleepMemorySnapshot("sleep/after_dereg_mr", local_rank);
        return true;
    };
    hooks.releaseKvMemoryBacking = [this, engine, local_rank](const SleepOptions&) {
        OptionalSleepDeviceGuard device_guard(local_rank);
        auto                     cache_manager = engine->getCacheManager();
        if (!cache_manager) {
            return true;
        }
        // Host memory-cache tier (enable_memory_cache): a pinned host buffer that is NOT
        // under any VMM tag and NOT MR-registered, so the VMM pause below cannot free it.
        // Discard it explicitly here (no-op when the memory cache is disabled). Runs on
        // every sleep level; in-flight H2D/D2H copies were already drained (connector_inflight).
        if (!cache_manager->releaseMemoryCacheBacking()) {
            RTP_LLM_LOG_WARNING("releaseKvMemoryBacking: releaseMemoryCacheBacking failed");
            return false;
        }
        auto controller = cache_manager->kvMemoryController();
        if (!controller || !controller->backendAvailable()) {
            RTP_LLM_LOG_WARNING("releaseKvMemoryBacking skipped VMM pause: backend unavailable");
            return true;
        }
        const bool success = cache_manager->releaseKVCacheMemoryBacking();
        // Memory-cache pinned host buffer + GPU KV released here: watch sys MemAvailable
        // (host pinned) rise and gpu_free rise.
        logSleepMemorySnapshot("sleep/after_kv_release", local_rank);
        return success;
    };
    // M6: weights are tagged by rtp_llm/model_loader/weight_memory_saver.py under
    // "weights" with cpu backup. CUDA graph runtime buffers are tagged under
    // "cuda_graph" during graph capture with cpu backup too: after VMM pause
    // the physical pages can be recycled by other processes, so graph-owned
    // persistent buffers cannot rely on stale physical contents. Releasing an
    // unknown tag is a harmless no-op.
    hooks.releaseRestorableGpuMemory = [this, vmm_backend, local_rank](const SleepOptions& opt) {
        OptionalSleepDeviceGuard device_guard(local_rank);
        if (!vmm_backend->isAvailable()) {
            RTP_LLM_LOG_WARNING("releaseRestorableGpuMemory skipped: VMM backend unavailable");
            return true;
        }
        auto pause_tag = [vmm_backend](const std::string& tag) { return vmm_backend->pause(tag); };
        // Level 2 (discard weights): the "weights" region was opened without host
        // cpu_backup, so pause frees GPU with no host copy and nothing is written
        // anywhere. Wake reloads the weights in place from the model loader
        // (see restoreRestorableGpuMemory below), so sleep just pauses the region.
        // These pauses are the ESSENTIAL GPU release; do them first so a failure in the
        // best-effort emptyCache() below can never leave the regions mapped.
        bool ok = pause_tag("cuda_graph");
        ok      = pause_tag("weights") && ok;
        // The engine is only paused (not torn down): the still-alive executor's transient
        // buffers (activations, attention/cuBLAS workspaces, sampler) can linger in the torch
        // device caching allocator as reserved-but-free blocks, which cudaMemGetInfo still
        // counts as used. Return them to the driver; the allocator transparently re-grows on
        // wake. Only frees FREE cached blocks; the VMM-tagged weights/kv/cuda_graph regions
        // paused above are still torch-"allocated", so this does not target them.
        //
        // BEST-EFFORT ONLY: on a decode role with captured CUDA graphs, the caching allocator
        // holds graph-private MemPool blocks backed by torch_memory_saver VMM. emptyCache()'s
        // release_block() then issues a cuMemUnmap/cudaFree that returns "CUDA error: invalid
        // argument" (reproduced on decode DP2 + CUDA graph regardless of whether it runs before
        // or after the pauses; prefill without graph never hits it -- same family as the
        // MemPool-destroy-under-TMS issue). This reclaim is non-essential -- yield is near-zero
        // once the engine is quiesced -- so a failure must NOT fail the sleep: swallow it, drain
        // the sticky CUDA error so it can't poison the subsequent wake, and continue.
#if USING_CUDA || USING_ROCM
        {
            OptionalSleepDeviceGuard empty_cache_guard(local_rank);
            try {
                c10::cuda::CUDACachingAllocator::emptyCache();
            } catch (const std::exception& e) {
                // clear sticky error left by the failed free so it can't poison the next wake
#if USING_CUDA
                (void)cudaGetLastError();
#elif USING_ROCM
                (void)hipGetLastError();
#endif
                RTP_LLM_LOG_WARNING("releaseRestorableGpuMemory: best-effort emptyCache() failed (%s); "
                                    "continuing, GPU regions already released via VMM pause",
                                    e.what());
            }
        }
#endif
        // Terminal sleep state: weights + cuda_graph GPU memory released (level-2 keeps no backup).
        logSleepMemorySnapshot("sleep/SLEEPING", local_rank);
        return ok;
    };
    hooks.restoreKvMemoryBackingAndResetMetadata = [this, engine, local_rank]() {
        OptionalSleepDeviceGuard device_guard(local_rank);
        auto                     cache_manager = engine->getCacheManager();
        if (!cache_manager) {
            return true;
        }
        // Reallocate the host memory-cache pinned buffer first (mirrors the sleep-side
        // release order). Independent of the GPU KV VMM resume below; no-op when disabled.
        if (!cache_manager->restoreMemoryCacheBacking()) {
            RTP_LLM_LOG_WARNING("restoreKvMemoryBackingAndResetMetadata: restoreMemoryCacheBacking failed");
            return false;
        }
        auto controller = cache_manager->kvMemoryController();
        if (!controller || !controller->isPaused()) {
            return true;  // pause was skipped (no shim); keep metadata untouched
        }
        // Re-maps pages at the same VA, then resets BlockPool metadata + BlockCache.
        const bool success = cache_manager->restoreKVCacheMemoryBackingAndResetMetadata();
        logSleepMemorySnapshot("wake/after_kv_restore", local_rank);
        return success;
    };
    hooks.restoreRestorableGpuMemory = [this, engine, vmm_backend, local_rank]() {
        OptionalSleepDeviceGuard device_guard(local_rank);
        if (!vmm_backend->isAvailable()) {
            return true;
        }
        auto resume_tag = [vmm_backend](const std::string& tag) { return vmm_backend->resume(tag); };
        // resume("weights") and resume("cuda_graph") remap physical pages at the same VA.
        // For level 1 the tms host cpu_backup already restored the content; for level 2 the
        // weights pages come back blank and are reloaded below.
        //
        // IMPORTANT (level 2): resume BOTH VMM regions BEFORE the loader reload. The reload
        // allocates transient shard/dequant buffers via the torch caching allocator
        // (at::empty_cuda). If the "cuda_graph" region is still paused at that point, the
        // allocator's graph-private MemPool is left with unmapped VMM pages, and a fresh
        // at::empty_cuda throws "CUDA error: invalid argument" (reproduced on decode DP2 +
        // CUDA graph; same MemPool-under-TMS family as the sleep-side emptyCache failure).
        // Resuming cuda_graph first re-maps that pool so the reload's allocations succeed.
        bool ok = resume_tag("weights");
        ok      = resume_tag("cuda_graph") && ok;
        if (ok && engine->sleepController().activeSleepLevel() == 2) {
            if (weight_manager_.is_none()) {
                RTP_LLM_LOG_WARNING("level-2 wake: weight_manager unavailable, cannot reload weights");
                return false;
            }
            try {
                py::gil_scoped_acquire acquire;
                weight_manager_.attr("reload_weights_from_loader")();
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_WARNING("level-2 wake: reload_weights_from_loader failed: %s", e.what());
                return false;
            }
        }
        // Weights (level-1 host restore / level-2 loader reload) + cuda_graph GPU memory back.
        logSleepMemorySnapshot("wake/after_weights_restore", local_rank);
        return ok;
    };
    hooks.registerMr = [this, engine, local_rank]() {
        OptionalSleepDeviceGuard device_guard(local_rank);
        if (!synchronizeSleepDevice("before_reg_mr")) {
            return false;
        }
        if (auto cache_manager = engine->getCacheManager()) {
            cache_manager->regUserMr(maga_init_params_.model_id, cache_manager->getCacheStore());
        }
        // Internal RDMA backends must publish refreshed rkey/lkey/epoch here
        // before the engine loop restarts. The open-source CacheStore path has
        // no peer-visible MR epoch ABI; regUserMr() is the available boundary.
        // Fully restored: MR re-registered, VmPin should be back at the baseline.
        logSleepMemorySnapshot("wake/RUNNING", local_rank);
        return true;
    };
    hooks.restartEngine = [engine]() {
        engine->restart();
        return true;
    };
    hooks.cancelQuiesceAndRestartEngine = hooks.restartEngine;
    hooks.warmupAndHealthCheck          = [this, engine, local_rank]() {
        OptionalSleepDeviceGuard device_guard(local_rank);
        if (!engine) {
            RTP_LLM_LOG_ERROR("sleep warmup/self-check failed: engine is null");
            return false;
        }
        if (!synchronizeSleepDevice("before_warmup_health_check")) {
            return false;
        }
        if (auto cache_manager = engine->getCacheManager()) {
            if (auto controller = cache_manager->kvMemoryController()) {
                if (controller->isPaused()) {
                    RTP_LLM_LOG_ERROR("sleep warmup/self-check failed: kv memory controller is still paused");
                    return false;
                }
                if (controller->basePtr() == nullptr || controller->totalSizeBytes() == 0) {
                    RTP_LLM_LOG_WARNING("sleep warmup/self-check: kv memory controller has no attached buffer");
                }
            }
        }
        if (!synchronizeSleepDevice("after_warmup_health_check")) {
            return false;
        }
        RTP_LLM_LOG_INFO("sleep warmup/self-check passed");
        return true;
    };

    engine_->sleepController().setHooks(hooks);
}

std::shared_ptr<void> LocalRpcServer::registerAbortableStreamForScope(const std::shared_ptr<GenerateStream>& stream) {
    if (!stream || stream->isStreaming()) {
        return nullptr;
    }
    const auto request_id = stream->streamId();
    {
        std::lock_guard<std::mutex> lock(abortable_streams_mutex_);
        abortable_streams_[request_id] = stream;
    }
    RTP_LLM_LOG_DEBUG("sleep abort registry: registered non-streaming request [%ld]", request_id);
    // Non-owning RAII token: the custom deleter only unregisters; it must not delete the stream.
    return std::shared_ptr<void>(stream.get(), [this, request_id](void*) { unregisterAbortableStream(request_id); });
}

void LocalRpcServer::unregisterAbortableStream(int64_t request_id) {
    std::lock_guard<std::mutex> lock(abortable_streams_mutex_);
    abortable_streams_.erase(request_id);
}

size_t LocalRpcServer::cancelAbortableStreams() {
    std::vector<std::pair<int64_t, std::shared_ptr<GenerateStream>>> streams;
    {
        std::lock_guard<std::mutex> lock(abortable_streams_mutex_);
        for (auto iter = abortable_streams_.begin(); iter != abortable_streams_.end();) {
            auto stream = iter->second.lock();
            if (!stream) {
                iter = abortable_streams_.erase(iter);
                continue;
            }
            streams.emplace_back(iter->first, std::move(stream));
            ++iter;
        }
    }

    size_t cancelled = 0;
    for (const auto& [request_id, stream] : streams) {
        if (!stream || stream->isStreaming() || stream->isFinished() || stream->hasError()) {
            continue;
        }
        stream->reportError(ErrorCode::CANCELLED, "request cancelled by sleep abort");
        RTP_LLM_LOG_WARNING("sleep abort registry: cancelled non-streaming request [%ld]", request_id);
        ++cancelled;
    }
    return cancelled;
}

grpc::Status LocalRpcServer::serializeErrorMsg(const string& request_key, ErrorInfo error_info) {
    const auto& error_msg = error_info.ToString();
    RTP_LLM_LOG_WARNING("request [%s], error code [%s], error message [%s]",
                        request_key.c_str(),
                        ErrorCodeToString(error_info.code()).c_str(),
                        error_msg.c_str());
    auto           grpc_error_code = transErrorCodeToGrpc(error_info.code());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_info.code()));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc_error_code, error_msg, error_details_serialized);
    } else {
        RTP_LLM_LOG_WARNING("request [%s] error details serialize to string failed", request_key.c_str());
        return grpc::Status(grpc_error_code, error_msg);
    }
}

grpc::Status LocalRpcServer::pollStreamOutput(grpc::ServerContext*             context,
                                              const string&                    request_key,
                                              WriterInterface*                 writer,
                                              std::shared_ptr<GenerateStream>& stream) {
    RTP_LLM_PROFILE_FUNCTION();
    // 需要检查 !hasError(): 之前 finished() 表示完成且无错，现在 FINISHED 状态可能包含错误
    // 如果流有错误，应该停止消费输出
    while (stream->isActive() || stream->hasOutput()) {
        const auto result = stream->nextOutput();
        if (!result.ok()) {
            if (result.status().code() != ErrorCode::FINISHED) {
                return serializeErrorMsg(request_key, result.status());
            } else {
                break;
            }
        }
        RTP_LLM_LOG_DEBUG("request [%s] generate next output success", request_key.c_str());
        GenerateOutputsPB outputs_pb;

        QueryConverter::transResponse(&outputs_pb,
                                      &(result.value()),
                                      stream->generateConfig()->aux_info,
                                      maga_init_params_.misc_config.aux_string,
                                      stream->specialTokens().eos_token_id);
        if (context->IsCancelled()) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by user");
            RTP_LLM_LOG_WARNING("request [%s] cancelled by user", request_key.c_str());
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!writer->Write(outputs_pb)) {
            stream->reportError(ErrorCode::CANCELLED, "write outputs pb failed");
            RTP_LLM_LOG_WARNING("request [%s] write outputs pb failed", request_key.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
        }
        if (stream->hasEvent(StreamEvents::NeedRemoteGenerate)) {
            break;
        }
        if (stream->queryPdSep()) {
            stream->waitForRemoteGenerate();
            break;
        }
    }
    RTP_LLM_LOG_DEBUG("request [%s] local generate done", request_key.c_str());

    return grpc::Status::OK;
}

ErrorInfo LocalRpcServer::prepareInput(const GenerateInputPB& input_pb, std::shared_ptr<GenerateInput>& output) {
    output = QueryConverter::transQuery(&input_pb);
    if (mm_processor_ != nullptr && output->multimodal_inputs) {
        RTP_LLM_PROFILE_SCOPE("rpc.mm_update_features");
        auto mm_res = mm_processor_->updateMultimodalFeatures(output);
        if (!mm_res.ok()) {
            return mm_res;
        }
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo LocalRpcServer::collectStreamOutput(grpc::ServerContext*                  context,
                                              std::shared_ptr<GenerateStream>&      stream,
                                              const std::shared_ptr<GenerateInput>& input,
                                              GenerateOutputs&                      last_outputs) {
    while (!stream->isFinished() || stream->hasOutput()) {
        if (context->IsCancelled()) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by client");
            return ErrorInfo(ErrorCode::CANCELLED, "request cancelled by client");
        }
        const auto output_result = stream->nextOutput();
        if (!output_result.ok()) {
            if (output_result.status().code() != ErrorCode::FINISHED) {
                return output_result.status();
            }
            break;
        }
        last_outputs = output_result.value();
    }
    return ErrorInfo::OkStatus();
}

grpc::Status LocalRpcServer::GenerateStreamCall(grpc::ServerContext*                   context,
                                                const GenerateInputPB*                 request,
                                                grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_SCOPE("rpc.generate_stream_call");
    auto admission = acquireAdmission();
    if (!admission.detail.admitted) {
        return AdmissionGate::toGrpcStatus(admission.detail);
    }
    auto               admission_lease = std::move(admission.lease);
    c10::InferenceMode inference_guard(true);
    AtomicGuard        request_guard(onflight_requests_);
    auto               request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), context, metrics_reporter_, meta_);
    std::shared_ptr<GenerateInput> input;
    {
        auto mm_res = prepareInput(*request, input);
        if (!mm_res.ok()) {
            generate_context.error_status = serializeErrorMsg(generate_context.request_key, mm_res);
        }
    }
    CHECK_ERROR_STATUS(generate_context);

    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    {
        RTP_LLM_PROFILE_SCOPE("rpc.enqueue_engine");
        generate_context.setStream(engine_->enqueue(input));
    }
    auto abort_registration = registerAbortableStreamForScope(generate_context.getStream());

    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", request_id);

    generate_context.error_status =
        pollStreamOutput(context, generate_context.request_key, writer, generate_context.getStream());
    meta_->dequeue(generate_context.request_id, generate_context.getStream());
    return generate_context.error_status;
}

grpc::Status LocalRpcServer::BatchGenerateCall(grpc::ServerContext*        context,
                                               const BatchGenerateInputPB* request,
                                               BatchGenerateOutputsPB*     response) {
    RTP_LLM_PROFILE_SCOPE("rpc.batch_generate_call");
    // Whole-batch rejection: a non-RUNNING instance must not run any of them.
    auto admission = acquireAdmission();
    if (!admission.detail.admitted) {
        return AdmissionGate::toGrpcStatus(admission.detail);
    }
    auto               admission_lease = std::move(admission.lease);
    c10::InferenceMode inference_guard(true);
    AtomicGuard        request_guard(onflight_requests_);
    const int          batch_size = request->inputs_size();
    RTP_LLM_LOG_INFO("receive batch generate request, batch_size=%d", batch_size);

    if (batch_size == 0) {
        return grpc::Status::OK;
    }

    std::vector<std::shared_ptr<GenerateInput>> inputs;
    inputs.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
        std::shared_ptr<GenerateInput> input;
        auto                           err = prepareInput(request->inputs(i), input);
        if (!err.ok()) {
            // Fill error results for all requests (0..batch_size-1) to maintain 1:1 mapping
            for (int j = 0; j < batch_size; j++) {
                auto* result = response->add_results();
                auto* err_pb = result->mutable_error_info();
                err_pb->set_error_code(ErrorCodePB::UNKNOWN_ERROR);
                if (j == i) {
                    err_pb->set_error_message("multimodal processing failed: " + err.ToString());
                } else {
                    err_pb->set_error_message("batch aborted due to multimodal failure at index " + std::to_string(i));
                }
            }
            return grpc::Status::OK;
        }
        inputs.push_back(input);
    }

    // batchEnqueue contract: returned vector is 1:1 with `inputs` (same size, same order).
    // Streams that failed checkInputLength carry an error reported via reportError() and surface
    // it through collectStreamOutput → nextOutput → ErrorInfo path below.
    auto                               streams = engine_->batchEnqueue(inputs);
    std::vector<std::shared_ptr<void>> abort_registrations;
    abort_registrations.reserve(streams.size());
    for (const auto& stream : streams) {
        abort_registrations.push_back(registerAbortableStreamForScope(stream));
    }

    // collectStreamOutput is currently SERIAL: streams[0] must finish before streams[1] is drained.
    // For batch decode this is bounded (all streams advance together), but TODO: parallelize for
    // mixed-length batches.
    for (int i = 0; i < (int)streams.size(); i++) {
        auto* result = response->add_results();

        GenerateOutputs last_outputs;
        auto            err = collectStreamOutput(context, streams[i], inputs[i], last_outputs);
        if (!err.ok()) {
            auto* err_pb = result->mutable_error_info();
            err_pb->set_error_code(err.code() == ErrorCode::CANCELLED ? ErrorCodePB::CANCELLED :
                                                                        ErrorCodePB::UNKNOWN_ERROR);
            err_pb->set_error_message(err.ToString());
        } else {
            auto* output_pb = result->mutable_final_output();
            QueryConverter::transResponse(output_pb,
                                          &last_outputs,
                                          inputs[i]->generate_config->aux_info,
                                          maga_init_params_.misc_config.aux_string,
                                          streams[i]->specialTokens().eos_token_id);
        }
    }

    RTP_LLM_LOG_INFO("batch generate done, batch_size=%d", batch_size);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::GetCacheStatus(grpc::ServerContext* context, const CacheVersionPB* request, CacheStatusPB* response) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s, request cache version: [%d]",
                      context->peer().c_str(),
                      request->latest_cache_version());
    KVCacheInfo cache_status = getCacheStatusInfo(request->latest_cache_version(), request->need_cache_keys());
    response->set_available_kv_cache(cache_status.available_kv_cache);
    response->set_total_kv_cache(cache_status.total_kv_cache);
    response->set_block_size(cache_status.block_size);
    response->set_version(cache_status.version);
    auto* cache_map = response->mutable_cache_keys();
    for (const auto& key : cache_status.cached_keys) {
        (*cache_map)[static_cast<int64_t>(key)] = true;
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::GetWorkerStatus(grpc::ServerContext*   context,
                                             const StatusVersionPB* request,
                                             WorkerStatusPB*        response) {
    RTP_LLM_PROFILE_FUNCTION();
    int64_t request_begin_time_us   = currentTimeUs();
    int64_t latest_finished_version = request->latest_finished_version();
    RTP_LLM_LOG_DEBUG(
        "receive workerStatus rpc request from client: %s, latest_finished_version: %ld, config role_type: %d",
        context->peer().c_str(),
        latest_finished_version,
        maga_init_params_.pd_sep_config.role_type);

    WorkerStatusInfo status_info              = getWorkerStatusInfo(latest_finished_version);
    int64_t          request_after_ws_time_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("getWorkerStatusInfo took %ld us", request_after_ws_time_us - request_begin_time_us);

    const auto& engine_schedule_info = status_info.engine_schedule_info;
    response->set_role(status_info.role);

    for (const auto& task : engine_schedule_info.running_task_info_list) {
        TaskInfoPB* task_info = response->add_running_task_info();
        task_info->set_request_id(task.request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_is_waiting(task.is_waiting);
    }

    for (const auto& task : engine_schedule_info.finished_task_info_list) {
        TaskInfoPB* task_info = response->add_finished_task_list();
        task_info->set_request_id(task.request_id);
        task_info->set_prefix_length(task.prefix_length);
        task_info->set_input_length(task.input_length);
        task_info->set_waiting_time_ms(task.waiting_time_ms);
        task_info->set_iterate_count(task.iterate_count);
        task_info->set_end_time_ms(task.end_time_ms);
        task_info->set_dp_rank(status_info.dp_rank);
        task_info->set_is_waiting(task.is_waiting);
    }
    response->set_dp_size(status_info.dp_size);
    response->set_tp_size(status_info.tp_size);
    response->set_status_version(status_info.status_version);
    response->set_latest_finished_version(status_info.latest_finished_version);
    response->set_alive(status_info.alive);
    response->set_precision(status_info.precision);
    reportWorkerStatusTime(request_begin_time_us, request_after_ws_time_us);
    return grpc::Status::OK;
}

WorkerStatusInfo LocalRpcServer::getWorkerStatusInfo(int64_t latest_finished_version) {
    WorkerStatusInfo status_info;
    status_info.engine_schedule_info = getEngineScheduleInfo(latest_finished_version);
    switch (maga_init_params_.pd_sep_config.role_type) {
        case RoleType::PDFUSION:
            status_info.role = "RoleType.PDFUSION";
            break;
        case RoleType::PREFILL:
            status_info.role = "RoleType.PREFILL";
            break;
        case RoleType::DECODE:
            status_info.role = "RoleType.DECODE";
            break;
        case RoleType::VIT:
            status_info.role = "RoleType.VIT";
            break;
        case RoleType::FRONTEND:
            status_info.role = "RoleType.FRONTEND";
            break;
        default:
            status_info.role = "RoleType.UNKNOWN";
    }
    status_info.dp_size                 = maga_init_params_.parallelism_config.dp_size;
    status_info.tp_size                 = maga_init_params_.parallelism_config.tp_size;
    status_info.dp_rank                 = maga_init_params_.parallelism_config.dp_rank;
    status_info.status_version          = currentTimeUs();
    status_info.latest_finished_version = status_info.engine_schedule_info.latest_finished_version;
    // Sleep takes the worker out of LB rotation. alive doubles as the
    // schedulable signal in WorkerStatusPB; non-RUNNING -> not schedulable.
    status_info.alive = engine_ ? engine_->sleepController().admit() : true;
    auto quant_method = maga_init_params_.model_config_.quant_algo.getQuantMethod();

    switch (quant_method) {
        case QuantMethod::WeightOnlyPerCol:
            status_info.precision = "WeightOnlyPerCol";
            break;
        case QuantMethod::GptQ:
            status_info.precision = "GptQ";
            break;
        case QuantMethod::Awq:
            status_info.precision = "Awq";
            break;
        case QuantMethod::SmoothQuant:
            status_info.precision = "SmoothQuant";
            break;
        case QuantMethod::OmniQuant:
            status_info.precision = "OmniQuant";
            break;
        case QuantMethod::PerTensorQuant:
            status_info.precision = "PerTensorQuant";
            break;
        case QuantMethod::FP8Quant:
            status_info.precision = "FP8Quant";
            break;
        case QuantMethod::FP8PTPC:
            status_info.precision = "FP8PTPC";
            break;
        case QuantMethod::W4A8INT4PTPC:
            status_info.precision = "W4A8INT4PTPC";
            break;
        case QuantMethod::None:
            status_info.precision = "FP16";
            break;
        default:
            RTP_LLM_LOG_ERROR("unknown quant method: %d", static_cast<int>(quant_method));
            status_info.precision = "UNKNOWN";
    }
    return status_info;
}

KVCacheInfo LocalRpcServer::getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) {
    int64_t     request_begin_time_us = currentTimeUs();
    const auto& cache_info            = engine_->getCacheStatusInfo(latest_version, need_cache_keys);
    reportCacheStatusTime(request_begin_time_us);
    return cache_info;
}

size_t LocalRpcServer::onflightRequestNum() {
    return onflight_requests_;
}

size_t LocalRpcServer::activeCacheTransferCount() {
    if (!engine_) {
        return 0;
    }
    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        return 0;
    }
    auto cache_store = cache_manager->getCacheStore();
    return cache_store ? cache_store->activeTransferCount() : 0;
}

EngineScheduleInfo LocalRpcServer::getEngineScheduleInfo(int64_t latest_finished_version) {
    EngineScheduleInfo                        info = meta_->getEngineScheduleInfo(latest_finished_version);
    std::vector<EngineScheduleInfo::TaskInfo> running_task_info_list = engine_->getScheduler().runningTaskList();
    for (auto& task_info : info.running_task_info_list) {
        for (auto& running_task : running_task_info_list) {
            if (task_info.request_id == running_task.request_id) {
                task_info.is_waiting = false;
            }
        }
    }
    auto last_schedule_time = engine_->getLastScheduleTime();
    // in case last_schedule_delta is negative
    info.last_schedule_delta =
        std::max((int64_t)0, autil::TimeUtility::currentTimeInMilliSeconds() - last_schedule_time);
    return info;
}

grpc::Status LocalRpcServer::UpdateSchedulerInfo(grpc::ServerContext*                context,
                                                 const UpdateSchedulerInfoRequestPB* request,
                                                 EmptyPB*                            response) {
    const std::string scheduler_info = request->scheduler_info();
    engine_->getScheduler().updateSchedulerInfo(scheduler_info);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::SetLogLevel(grpc::ServerContext* context, const SetLogLevelRequestPB* request, EmptyPB* response) {
    std::string log_level_str = request->log_level();
    uint32_t    log_level     = alog::LOG_LEVEL_INFO;
    if (log_level_str == "INFO" || log_level_str == "info") {
        log_level = alog::LOG_LEVEL_INFO;
    } else if (log_level_str == "WARNING" || log_level_str == "warning") {
        log_level = alog::LOG_LEVEL_WARN;
    } else if (log_level_str == "DEBUG" || log_level_str == "debug") {
        log_level = alog::LOG_LEVEL_DEBUG;
    } else if (log_level_str == "TRACE" || log_level_str == "trace") {
        log_level = alog::LOG_LEVEL_TRACE1;
    } else {
        RTP_LLM_LOG_WARNING("set log level failed, unknown log level: %s", log_level_str.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid log_level format");
    }
    auto& logger = rtp_llm::Logger::getEngineLogger();
    logger.setBaseLevel(log_level);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::StartProfile(grpc::ServerContext* context, const StartProfileRequestPB* request, EmptyPB* response) {
    (void)response;
    RTP_LLM_LOG_INFO("start_profile from %s start_step=%d num_steps=%d enable_all_rank=%d",
                     context->peer().c_str(),
                     request->start_step(),
                     request->num_steps(),
                     int(request->enable_all_rank()));
    if (!request->enable_all_rank()) {
        engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
        return grpc::Status::OK;
    }
    if (maga_init_params_.parallelism_config.tp_rank != 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "enable_all_rank start_profile must be sent to tp_rank 0");
    }
    if (!profile_broadcaster_) {
        if (maga_init_params_.parallelism_config.tp_size <= 1) {
            RTP_LLM_LOG_INFO("start_profile enable_all_rank with tp_size=1, fallback to local start");
            engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
            return grpc::Status::OK;
        }
        return grpc::Status(grpc::StatusCode::INTERNAL, "tp broadcaster unavailable for enable_all_rank start_profile");
    }

    std::vector<StartProfileInternalRequestPB> requests(profile_broadcaster_->workerNum());
    for (auto& internal_request : requests) {
        internal_request.set_trace_name(request->trace_name());
        internal_request.set_start_step(request->start_step());
        internal_request.set_num_steps(request->num_steps());
    }
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const StartProfileInternalRequestPB&        internal_request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncStartProfileInternal(context.get(), internal_request, completion_queue);
    };
    auto broadcast_result = profile_broadcaster_->broadcast<StartProfileInternalRequestPB, EmptyPB>(
        requests, /*timeout_ms=*/3000, rpc_call);
    if (!broadcast_result) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to broadcast start_profile_internal to tp group");
    }
    broadcast_result->waitDone();
    if (!broadcast_result->success()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "broadcast start_profile_internal to tp group failed");
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::StartProfileInternal(grpc::ServerContext*                 context,
                                                  const StartProfileInternalRequestPB* request,
                                                  EmptyPB*                             response) {
    (void)response;
    RTP_LLM_LOG_INFO("start_profile_internal from %s start_step=%d num_steps=%d",
                     context->peer().c_str(),
                     request->start_step(),
                     request->num_steps());
    engine_->startTimelineProfiling(request->trace_name(), request->start_step(), request->num_steps());
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::CheckHealth(grpc::ServerContext* context, const EmptyPB* request, CheckHealthResponsePB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    // A sleeping or transitioning instance is not ready; report the sleep state and
    // a retryable UNAVAILABLE so LB health checks take it out of rotation.
    if (auto admission = checkAdmission(); !admission.ok()) {
        response->set_health(sleepStateToString(engine_->sleepController().state()));
        return admission;
    }
    response->set_health("OK");
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::UpdateEplbConfig(grpc::ServerContext*             context,
                                              const UpdateEplbConfigRequestPB* request,
                                              EmptyPB*                         response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    const string mode_str = request->mode();
    EPLBConfig   config;
    if (mode_str == "EPLB" || mode_str == "eplb") {
        config.eplb_mode = EplbMode::EPLB;
    } else if (mode_str == "STATS" || mode_str == "stats") {
        config.eplb_mode = EplbMode::STATS;
    } else if (mode_str == "NONE" || mode_str == "none") {
        config.eplb_mode = EplbMode::NONE;
    } else if (mode_str == "ALL" || mode_str == "all") {
        config.eplb_mode = EplbMode::ALL;
    } else {
        RTP_LLM_LOG_WARNING("set eplb mode failed, unknown mode : %s", mode_str.c_str());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid eplb mode");
    }
    config.eplb_update_time = request->update_time();
    engine_->updateEplbConfig(config);
    return grpc::Status::OK;
}

void LocalRpcServer::reportWorkerStatusTime(int64_t request_begin_time_us, int64_t request_after_ws_time_us) {
    RpcWorkerStatusMetricsCollector collector;
    collector.qps         = true;
    collector.total_rt_us = request_after_ws_time_us - request_begin_time_us;
    if (metrics_reporter_) {
        metrics_reporter_->report<RpcWorkerStatusMetrics, RpcWorkerStatusMetricsCollector>(nullptr, &collector);
    }
}

void LocalRpcServer::reportCacheStatusTime(int64_t request_begin_time_us) {
    RpcCacheStatusMetricsCollector collector;
    collector.qps         = true;
    collector.total_rt_us = (currentTimeUs() - request_begin_time_us);
    if (metrics_reporter_) {
        metrics_reporter_->report<RpcCacheStatusMetrics, RpcCacheStatusMetricsCollector>(nullptr, &collector);
    }
}

::grpc::Status LocalRpcServer::ExecuteFunction(::grpc::ServerContext*     context,
                                               const ::FunctionRequestPB* request,
                                               ::FunctionResponsePB*      response) {
    RTP_LLM_LOG_DEBUG("receive execute function request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());
    if (context->IsCancelled()) {
        RTP_LLM_LOG_WARNING("execute function failed, request is cancelled");
        return grpc::Status(grpc::StatusCode::CANCELLED, "request is cancelled");
    }
    if (!engine_) {
        RTP_LLM_LOG_WARNING("execute function failed, engine is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "engine is null");
    }

    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_LOG_WARNING("execute function failed, cache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "cache manager is null");
    }
    if (!cache_manager->executeFunction(*request, *response)) {
        RTP_LLM_LOG_WARNING("execute function failed, request: [%s]", request->DebugString().c_str());
        const std::string error_msg = "execute function failed, request: [" + request->DebugString() + "]";
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::SetPause(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s", context->peer().c_str());
    OptionalSleepDeviceGuard device_guard(maga_init_params_.parallelism_config.local_rank);
    auto                     status = engine_->pauseAndWaitQuiesced(60000);
    if (!status.ok()) {
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, status.ToString());
    }
    return grpc::Status::OK;
}

grpc::Status LocalRpcServer::SetRestart(grpc::ServerContext* context, const EmptyPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("receive cacheStatus rpc request from client: %s,", context->peer().c_str());
    engine_->restart();
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::SleepServing(grpc::ServerContext* context, const SleepRequestPB* request, EmptyPB* response) {
    RTP_LLM_LOG_INFO("receive SleepServing rpc request from client: %s, level: %d, mode: %s, reason: %s, "
                     "prepare_only: %d, commit_only: %d",
                     context->peer().c_str(),
                     request->level(),
                     request->mode().c_str(),
                     request->reason().c_str(),
                     request->prepare_only(),
                     request->commit_only());
    SleepOptions options;
    options.level        = request->level();
    options.mode         = request->mode().empty() ? "wait" : request->mode();
    options.timeout_ms   = request->timeout_ms();
    options.reason       = request->reason();
    options.tags         = std::vector<std::string>(request->tags().begin(), request->tags().end());
    options.prepare_only = request->prepare_only();
    options.commit_only  = request->commit_only();
    const auto result    = engine_->sleepController().sleep(options);
    return sleepResultToGrpcStatus(result);
}

grpc::Status
LocalRpcServer::WakeUpServing(grpc::ServerContext* context, const WakeUpRequestPB* request, EmptyPB* response) {
    RTP_LLM_LOG_INFO("receive WakeUpServing rpc request from client: %s, prepare_only: %d, commit_only: %d",
                     context->peer().c_str(),
                     request->prepare_only(),
                     request->commit_only());
    WakeUpOptions options;
    options.prepare_only = request->prepare_only();
    options.commit_only  = request->commit_only();
    const auto result    = engine_->sleepController().wakeUp(options);
    return sleepResultToGrpcStatus(result);
}

grpc::Status
LocalRpcServer::IsSleeping(grpc::ServerContext* context, const EmptyPB* request, IsSleepingResponsePB* response) {
    RTP_LLM_LOG_DEBUG("receive IsSleeping rpc request from client: %s", context->peer().c_str());
    const auto status = engine_->sleepController().status();
    response->set_is_sleeping(status.state == SleepState::SLEEPING);
    response->set_sleep_mode_enabled(status.sleep_mode_enabled);
    response->set_effective(status.effective);
    response->set_state(sleepStateToString(status.state));
    response->set_disabled_reason(status.disabled_reason);
    for (const auto level : status.supported_levels) {
        response->add_supported_levels(level);
    }
    for (const auto& mode : status.supported_modes) {
        response->add_supported_modes(mode);
    }
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::GetSleepStatus(grpc::ServerContext* context, const EmptyPB* request, SleepStatusResponsePB* response) {
    RTP_LLM_LOG_DEBUG("receive GetSleepStatus rpc request from client: %s", context->peer().c_str());
    const auto status = engine_->sleepController().status();
    fillSleepStatusPb(status, response);
    return grpc::Status::OK;
}

grpc::Status
LocalRpcServer::UpdateWeights(grpc::ServerContext* context, const UpdateWeightsRequestPB* request, EmptyPB* response) {
    RTP_LLM_LOG_DEBUG("Receive update weights request from: %s", context->peer().c_str());
    try {
        if (request->name().empty() || request->desc().empty() || request->method().empty()) {
            throw std::runtime_error("Missing required field(s) in request");
        }
        {
            py::gil_scoped_acquire acquire;
            py::dict               req;
            req["name"]   = request->name();
            req["desc"]   = request->desc();
            req["method"] = request->method();
            weight_manager_.attr("update")(req);
        }
        return grpc::Status::OK;
    } catch (const py::error_already_set& e) {
        PyObject *type, *value, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        std::string err_msg = value ? PyUnicode_AsUTF8(value) : "Unknown Python error";
        return {grpc::StatusCode::INTERNAL, "exception from python: " + err_msg};
    } catch (const std::exception& e) {
        return {grpc::StatusCode::INTERNAL, "exception from C++: " + std::string(e.what())};
    }
}
}  // namespace rtp_llm
