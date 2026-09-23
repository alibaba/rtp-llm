#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/common/kernels/sm_copy_kernel.h"
#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/utils/CudacoreDiagnostics.h"
#include "rtp_llm/cpp/utils/CudacoreFlightRecorder.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace rtp_llm {

namespace {

at::cuda::CUDAStream& getNoBlockCopyStream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    return stream;
}

enum class HostCoverage {
    Invalid,
    Partial,
    Full,
};

std::mutex& cudaBatchSubmitMutex(int device_index) {
    static std::mutex                                           registry_mutex;
    static std::unordered_map<int, std::unique_ptr<std::mutex>> mutexes;

    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    auto&                       mutex = mutexes[device_index];
    if (!mutex) {
        mutex = std::make_unique<std::mutex>();
    }
    return *mutex;
}

HostCoverage checkHostCoverage(const StagedMemoryCopyParams& params) {
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(params.tiles.size());
    for (const auto& tile : params.tiles) {
        if (tile.bytes == 0) {
            continue;
        }
        if (tile.host_offset > params.host_bytes || tile.bytes > params.host_bytes - tile.host_offset) {
            return HostCoverage::Invalid;
        }
        ranges.emplace_back(tile.host_offset, tile.bytes);
    }
    std::sort(ranges.begin(), ranges.end());

    size_t covered = 0;
    bool   has_gap = false;
    for (const auto& [offset, bytes] : ranges) {
        if (bytes == 0 || offset < covered) {
            return HostCoverage::Invalid;
        }
        if (offset > covered) {
            has_gap = true;
        }
        covered = offset + bytes;
    }
    if (covered > params.host_bytes) {
        return HostCoverage::Invalid;
    }
    return (!has_gap && covered == params.host_bytes) ? HostCoverage::Full : HostCoverage::Partial;
}

bool checkHostSegments(const StagedMemoryCopyParams& params) {
    if (params.host_segments.empty()) {
        return params.host_base != nullptr && params.host_bytes > 0;
    }

    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(params.host_segments.size());
    for (const auto& segment : params.host_segments) {
        if (segment.host == nullptr || segment.bytes == 0) {
            return false;
        }
        if (segment.host_offset > params.host_bytes || segment.bytes > params.host_bytes - segment.host_offset) {
            return false;
        }
        ranges.emplace_back(segment.host_offset, segment.bytes);
    }
    std::sort(ranges.begin(), ranges.end());

    size_t covered = 0;
    for (const auto& [offset, bytes] : ranges) {
        if (offset < covered) {
            return false;
        }
        covered = offset + bytes;
    }
    return covered <= params.host_bytes;
}

void packHostSegments(const StagedMemoryCopyParams& params, void* host_staging) {
    auto* base = static_cast<char*>(host_staging);
    for (const auto& segment : params.host_segments) {
        std::memcpy(base + segment.host_offset, segment.host, segment.bytes);
    }
}

void unpackHostSegments(const StagedMemoryCopyParams& params, const void* host_staging) {
    const auto* base = static_cast<const char*>(host_staging);
    for (const auto& segment : params.host_segments) {
        std::memcpy(segment.host, base + segment.host_offset, segment.bytes);
    }
}

void copyHostToPinnedStaging(const StagedMemoryCopyParams& params, void* host_staging) {
    if (params.host_segments.empty()) {
        std::memcpy(host_staging, params.host_base, params.host_bytes);
        return;
    }
    packHostSegments(params, host_staging);
}

void copyPinnedStagingToHost(const StagedMemoryCopyParams& params, const void* host_staging) {
    if (params.host_segments.empty()) {
        std::memcpy(params.host_base, host_staging, params.host_bytes);
        return;
    }
    unpackHostSegments(params, host_staging);
}

void releaseDevicePointer(void*& ptr) {
    if (ptr != nullptr) {
        (void)cudaFree(ptr);
        ptr = nullptr;
    }
}

void releaseMetadataScratch(StagedMemoryCopyScratch& scratch) {
    releaseDevicePointer(scratch.device_ptrs);
    releaseDevicePointer(scratch.device_offsets);
    releaseDevicePointer(scratch.device_sizes);
    scratch.meta_capacity = 0;
}

// Copies complete tile metadata for a fatal device error before throwing or
// falling back. Ordinary OOM / argument errors stay out of this path.
void recordFatalCopyError(cudaError_t        error,
                          FatalCudaErrorSite site,
                          const char*        file,
                          int                line,
                          int                device_index,
                          cudaStream_t       stream,
                          void* const*       dsts,
                          const void* const* srcs,
                          const size_t*      sizes,
                          size_t             count,
                          const char*        operation         = "",
                          const std::string& operation_context = {}) {
    if (!isFatalCudaRuntimeError(static_cast<int>(error))) {
        return;
    }
    FatalCudaErrorRecord record = buildCudaRuntimeErrorRecord(static_cast<int>(error), site, file, line, device_index);
    record.message              = cudaGetErrorString(error);
    record.operation            = operation;
    record.operation_context_json = operation_context;
    char stream_repr[32]          = {};
    std::snprintf(stream_repr, sizeof(stream_repr), "%p", static_cast<const void*>(stream));
    record.stream = stream_repr;

    uint64_t bytes_total = 0;
    // Full evidence is written only on error. The incident summary applies its
    // own cap later; do not discard tiles before the supplemental file is saved.
    const size_t kept = count;
    record.tiles.reserve(kept);
    for (size_t index = 0; index < count; ++index) {
        const size_t bytes = sizes == nullptr ? 0 : sizes[index];
        bytes_total += bytes;
        if (index < kept) {
            FatalCudaTileRecord tile;
            tile.dst   = dsts == nullptr ? 0 : reinterpret_cast<uintptr_t>(dsts[index]);
            tile.src   = srcs == nullptr ? 0 : reinterpret_cast<uintptr_t>(srcs[index]);
            tile.bytes = bytes;
            record.tiles.push_back(tile);
        }
    }
    record.tile_total  = count;
    record.bytes_total = bytes_total;
    (void)recordFirstFatalCudaError(std::move(record));
}

StagedMemoryCopyStatus
ensureStagedMemoryCopyScratch(StagedMemoryCopyScratch& scratch, int device_index, size_t host_bytes, size_t tile_num) {
    if (scratch.device_index >= 0 && scratch.device_index != device_index) {
        releaseStagedMemoryCopyScratch(scratch);
    }
    check_cuda_value(cudaSetDevice(device_index));
    scratch.device_index = device_index;

    if (scratch.host_capacity < host_bytes) {
        if (scratch.host_staging != nullptr) {
            (void)cudaFreeHost(scratch.host_staging);
            scratch.host_staging  = nullptr;
            scratch.host_capacity = 0;
        }
        auto err = cudaHostAlloc(&scratch.host_staging, host_bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            recordFatalCopyError(err,
                                 FatalCudaErrorSite::StagedCopy,
                                 __FILE__,
                                 __LINE__,
                                 device_index,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 0,
                                 "cudaHostAlloc_staging");
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed to allocate pinned host staging: %s",
                                cudaGetErrorString(err));
            return err == cudaErrorMemoryAllocation ? StagedMemoryCopyStatus::RESOURCE_EXHAUSTED :
                                                      StagedMemoryCopyStatus::EXECUTION_FAILED;
        }
        scratch.host_capacity = host_bytes;
    }

    if (scratch.device_capacity < host_bytes) {
        releaseDevicePointer(scratch.device_staging);
        auto err = cudaMalloc(&scratch.device_staging, host_bytes);
        if (err != cudaSuccess) {
            scratch.device_capacity = 0;
            recordFatalCopyError(err,
                                 FatalCudaErrorSite::StagedCopy,
                                 __FILE__,
                                 __LINE__,
                                 device_index,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 0,
                                 "cudaMalloc_staging");
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed to allocate device staging: %s", cudaGetErrorString(err));
            return err == cudaErrorMemoryAllocation ? StagedMemoryCopyStatus::RESOURCE_EXHAUSTED :
                                                      StagedMemoryCopyStatus::EXECUTION_FAILED;
        }
        scratch.device_capacity = host_bytes;
    }

    if (scratch.meta_capacity < tile_num) {
        releaseMetadataScratch(scratch);
        const char* allocation = "cudaMalloc_device_ptrs";
        auto        err        = cudaMalloc(&scratch.device_ptrs, tile_num * sizeof(void*));
        if (err == cudaSuccess) {
            allocation = "cudaMalloc_device_offsets";
            err        = cudaMalloc(&scratch.device_offsets, tile_num * sizeof(size_t));
        }
        if (err == cudaSuccess) {
            allocation = "cudaMalloc_device_sizes";
            err        = cudaMalloc(&scratch.device_sizes, tile_num * sizeof(size_t));
        }
        if (err != cudaSuccess) {
            recordFatalCopyError(err,
                                 FatalCudaErrorSite::StagedCopy,
                                 __FILE__,
                                 __LINE__,
                                 device_index,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 0,
                                 allocation);
            if (!fatalCudacoreIncidentActive()) {
                releaseMetadataScratch(scratch);
            }
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed to allocate device metadata: %s", cudaGetErrorString(err));
            return err == cudaErrorMemoryAllocation ? StagedMemoryCopyStatus::RESOURCE_EXHAUSTED :
                                                      StagedMemoryCopyStatus::EXECUTION_FAILED;
        }
        scratch.meta_capacity = tile_num;
    }
    return StagedMemoryCopyStatus::SUCCESS;
}

}  // namespace

void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch& scratch) {
    if (scratch.device_index >= 0) {
        (void)cudaSetDevice(scratch.device_index);
    }
    if (scratch.host_staging != nullptr) {
        (void)cudaFreeHost(scratch.host_staging);
    }
    releaseDevicePointer(scratch.device_staging);
    releaseMetadataScratch(scratch);
    scratch.host_staging    = nullptr;
    scratch.host_capacity   = 0;
    scratch.device_capacity = 0;
    scratch.device_index    = -1;
}

void execNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    const bool has_cuda_tensor =
        !params.multi_dst.empty() && (params.multi_dst[0].is_cuda() || params.multi_src[0].is_cuda());
    const int copy_device =
        params.multi_dst.empty() ? getCopyDevice(-1, -1) : getCopyDevice(params.multi_dst[0], params.multi_src[0]);
    c10::cuda::CUDAGuard device_guard(copy_device);

    auto stream = getNoBlockCopyStream(copy_device).stream();

    if (params.split_kv_layer_num > 0 && has_cuda_tensor) {
        if (splitKvMultiCopy(params.multi_src,
                             params.multi_dst,
                             params.split_kv_layer_num,
                             static_cast<int64_t>(params.split_kv_cache_stride_bytes),
                             static_cast<int64_t>(params.split_kv_scale_stride_bytes),
                             stream)) {
            check_cuda_value(cudaStreamSynchronize(stream));
            check_cuda_error();
            return;
        }
    }

    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        check_cuda_value(cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                                         params.multi_src[i].data_ptr(),
                                         params.multi_src[i].nbytes(),
                                         cudaMemcpyDefault,
                                         stream));
    }
    check_cuda_value(cudaStreamSynchronize(stream));
    check_cuda_error();
}

BatchedMemoryCopyStatus execBatchedMemoryCopy(const BatchedMemoryCopyParams& params) {
    static const bool terminate_guard_installed = [] {
        installCudacoreTerminateGuard();
        return true;
    }();
    (void)terminate_guard_installed;
    if (params.tiles.empty()) {
        return BatchedMemoryCopyStatus::SUCCESS;
    }
    if (params.device_index < 0) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy failed: invalid device_index=%d", params.device_index);
        return BatchedMemoryCopyStatus::EXECUTION_FAILED;
    }

#if CUDART_VERSION >= 12080
    constexpr int min_batch_driver_version = CUDART_VERSION >= 13000 ? 13000 : 12080;
    const int     driver_version           = getCudaVersion();
    if (driver_version < min_batch_driver_version) {
        static std::once_flag driver_warning_once;
        std::call_once(driver_warning_once, [driver_version, min_batch_driver_version] {
            RTP_LLM_LOG_WARNING(
                "execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d requires driver API version "
                ">=%d for its cudaMemcpyBatchAsync ABI, installed driver API version=%d; falling back to generic "
                "copy",
                CUDART_VERSION,
                min_batch_driver_version,
                driver_version);
        });
        return BatchedMemoryCopyStatus::NOT_SUPPORTED;
    }

    int        runtime_version       = 0;
    const auto runtime_version_error = cudaRuntimeGetVersion(&runtime_version);
    if (runtime_version_error != cudaSuccess) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d, failed to query "
                            "runtime version (%s); cannot prove cudaMemcpyBatchAsync ABI compatibility",
                            CUDART_VERSION,
                            cudaGetErrorString(runtime_version_error));
        return BatchedMemoryCopyStatus::NOT_SUPPORTED;
    }
    if (runtime_version < 12080) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d, runtime version=%d "
                            "predates cudaMemcpyBatchAsync; falling back to generic copy",
                            CUDART_VERSION,
                            runtime_version);
        return BatchedMemoryCopyStatus::NOT_SUPPORTED;
    }
    const bool compiled_with_cuda13_batch_abi = CUDART_VERSION >= 13000;
    const bool runtime_uses_cuda13_batch_abi  = runtime_version >= 13000;
    if (compiled_with_cuda13_batch_abi != runtime_uses_cuda13_batch_abi) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d and runtime version=%d "
                            "use incompatible cudaMemcpyBatchAsync signatures; falling back to generic copy",
                            CUDART_VERSION,
                            runtime_version);
        return BatchedMemoryCopyStatus::NOT_SUPPORTED;
    }

    check_cuda_value(cudaSetDevice(params.device_index));
    auto stream = getNoBlockCopyStream().stream();

    const size_t             tile_num = params.tiles.size();
    std::vector<void*>       dsts;
    std::vector<const void*> srcs;
    std::vector<size_t>      sizes;
    dsts.reserve(tile_num);
    srcs.reserve(tile_num);
    sizes.reserve(tile_num);
    uint64_t total_bytes = 0;
    for (const auto& tile : params.tiles) {
        if (tile.dst == nullptr || tile.src == nullptr || tile.bytes == 0) {
            continue;
        }
        dsts.push_back(tile.dst);
        srcs.push_back(tile.src);
        sizes.push_back(tile.bytes);
        total_bytes += tile.bytes;
    }
    if (dsts.empty()) {
        return BatchedMemoryCopyStatus::SUCCESS;
    }

    CudacoreFlightEvent flight;
    flight.kind         = CudacoreFlightKind::BatchSubmit;
    flight.copy_id      = currentCudacoreCopyId();
    flight.device_index = params.device_index;
    flight.stream       = reinterpret_cast<uintptr_t>(stream);
    flight.tile_count   = dsts.size();
    flight.total_bytes  = total_bytes;
    flight.first_dst    = reinterpret_cast<uintptr_t>(dsts.front());
    flight.first_src    = reinterpret_cast<uintptr_t>(srcs.front());
    flight.first_bytes  = sizes.front();
    flight.last_dst     = reinterpret_cast<uintptr_t>(dsts.back());
    flight.last_src     = reinterpret_cast<uintptr_t>(srcs.back());
    flight.last_bytes   = sizes.back();
    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy phase=submit device=%d stream=%p tiles=%zu",
                      params.device_index,
                      static_cast<void*>(stream),
                      dsts.size());

    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx     = 0;
    // cuMemcpyBatchAsync_v2 in the deployed CUDA stack is not safe under
    // concurrent host submissions. Protect only the runtime API entry; each
    // call keeps its own stream and completion wait, so transfers may overlap.
    std::unique_lock<std::mutex> submit_lock(cudaBatchSubmitMutex(params.device_index));
    const uint64_t               submit_sequence = recordCudacoreFlightEvent(flight);
#if CUDART_VERSION >= 13000
    const auto submit_error =
        cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, stream);
#else
    std::vector<void*> mutable_srcs;
    mutable_srcs.reserve(srcs.size());
    for (auto* src : srcs) {
        mutable_srcs.push_back(const_cast<void*>(src));
    }
    size_t     fail_idx     = 0;
    const auto submit_error = cudaMemcpyBatchAsync(
        dsts.data(), mutable_srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, &fail_idx, stream);
#endif
    submit_lock.unlock();
    // Serialize only after an error; these versions/attributes are already
    // available on the host, so this adds no post-fault CUDA query.
    const auto batch_context = [&](bool submitting) {
        std::string json = "{\"compiled_cudart_version\":" + std::to_string(CUDART_VERSION);
        json += ",\"runtime_version\":" + std::to_string(runtime_version);
        json += ",\"driver_api_version\":" + std::to_string(driver_version);
        json += ",\"src_access_order\":" + std::to_string(static_cast<int>(attr.srcAccessOrder));
        json += ",\"attribute_index\":" + std::to_string(attr_idx);
        json += ",\"num_attributes\":1,\"other_attributes_zero_initialized\":true";
        json += ",\"submit_sequence\":" + std::to_string(submit_sequence);
        json += ",\"fail_index\":";
#if CUDART_VERSION < 13000
        // CUDA 12 exposes this output for failed batch submission only.
        json += submitting ? std::to_string(fail_idx) : "null";
#else
        (void)submitting;
        json += "null";
#endif
        return json + '}';
    };
    if (submit_error != cudaSuccess) {
        flight.kind             = CudacoreFlightKind::BatchSubmitResult;
        flight.related_sequence = submit_sequence;
        flight.cuda_error       = static_cast<int>(submit_error);
        (void)recordCudacoreFlightEvent(flight);
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy failed phase=submit device=%d stream=%p tiles=%zu error_code=%d "
                            "error=%s",
                            params.device_index,
                            static_cast<void*>(stream),
                            dsts.size(),
                            static_cast<int>(submit_error),
                            cudaGetErrorString(submit_error));
        recordFatalCopyError(submit_error,
                             FatalCudaErrorSite::BatchedCopySubmit,
                             __FILE__,
                             __LINE__,
                             params.device_index,
                             stream,
                             dsts.data(),
                             srcs.data(),
                             sizes.data(),
                             dsts.size(),
                             "cudaMemcpyBatchAsync",
                             batch_context(true));
        return BatchedMemoryCopyStatus::EXECUTION_FAILED;
    }

    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy phase=submitted device=%d stream=%p tiles=%zu",
                      params.device_index,
                      static_cast<void*>(stream),
                      dsts.size());

    const auto completion_error = cudaStreamSynchronize(stream);
    flight.kind                 = CudacoreFlightKind::BatchCompletion;
    flight.related_sequence     = submit_sequence;
    flight.cuda_error           = static_cast<int>(completion_error);
    (void)recordCudacoreFlightEvent(flight);
    if (completion_error != cudaSuccess) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy failed phase=completion device=%d stream=%p tiles=%zu "
                            "error_code=%d error=%s",
                            params.device_index,
                            static_cast<void*>(stream),
                            dsts.size(),
                            static_cast<int>(completion_error),
                            cudaGetErrorString(completion_error));
        recordFatalCopyError(completion_error,
                             FatalCudaErrorSite::BatchedCopyCompletion,
                             __FILE__,
                             __LINE__,
                             params.device_index,
                             stream,
                             dsts.data(),
                             srcs.data(),
                             sizes.data(),
                             dsts.size(),
                             "cudaStreamSynchronize",
                             batch_context(false));
        return BatchedMemoryCopyStatus::EXECUTION_FAILED;
    }
    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy phase=completed device=%d stream=%p tiles=%zu",
                      params.device_index,
                      static_cast<void*>(stream),
                      dsts.size());
    // cudaStreamSynchronize already reports deferred errors from this batch.
    // Do not call check_cuda_error() here: in DEBUG mode it performs a
    // device-wide synchronize and can wait on unrelated TP/NCCL work.
    if (Logger::getEngineLogger().isDebugMode()) {
        check_cuda_value(cudaGetLastError());
    }
    return BatchedMemoryCopyStatus::SUCCESS;
#else
    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy unavailable: CUDART_VERSION=%d", CUDART_VERSION);
    return BatchedMemoryCopyStatus::NOT_SUPPORTED;
#endif
}

StagedMemoryCopyStatus execStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch* scratch) {
    if (params.tiles.empty()) {
        return StagedMemoryCopyStatus::SUCCESS;
    }
    if (params.device_index < 0 || params.host_bytes == 0 || !checkHostSegments(params)) {
        RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: device=%d host_base=%p host_bytes=%zu host_segments=%zu",
                            params.device_index,
                            params.host_base,
                            params.host_bytes,
                            params.host_segments.size());
        return StagedMemoryCopyStatus::INVALID_ARGUMENT;
    }
    const auto host_coverage = checkHostCoverage(params);
    if (host_coverage == HostCoverage::Invalid) {
        RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: invalid/overlapping host coverage, tiles=%zu bytes=%zu",
                            params.tiles.size(),
                            params.host_bytes);
        return StagedMemoryCopyStatus::INVALID_ARGUMENT;
    }

    std::vector<void*>  h_ptrs;
    std::vector<size_t> h_offsets;
    std::vector<size_t> h_sizes;
    h_ptrs.reserve(params.tiles.size());
    h_offsets.reserve(params.tiles.size());
    h_sizes.reserve(params.tiles.size());
    for (const auto& tile : params.tiles) {
        if (tile.gpu == nullptr || tile.bytes == 0) {
            return StagedMemoryCopyStatus::INVALID_ARGUMENT;
        }
        if (tile.host_offset > params.host_bytes || tile.bytes > params.host_bytes - tile.host_offset) {
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: tile out of host span, off=%zu bytes=%zu host=%zu",
                                tile.host_offset,
                                tile.bytes,
                                params.host_bytes);
            return StagedMemoryCopyStatus::INVALID_ARGUMENT;
        }
        h_ptrs.push_back(tile.gpu);
        h_offsets.push_back(tile.host_offset);
        h_sizes.push_back(tile.bytes);
    }
    if (h_ptrs.empty()) {
        return StagedMemoryCopyStatus::SUCCESS;
    }

    check_cuda_value(cudaSetDevice(params.device_index));
    auto stream = getNoBlockCopyStream().stream();

    StagedMemoryCopyScratch local_scratch;
    auto*                   work_scratch          = scratch != nullptr ? scratch : &local_scratch;
    auto                    cleanup_local_scratch = [&]() {
        if (scratch == nullptr && !fatalCudacoreIncidentActive()) {
            releaseStagedMemoryCopyScratch(local_scratch);
        }
    };

    const size_t tile_num = h_ptrs.size();
    const auto   scratch_status =
        ensureStagedMemoryCopyScratch(*work_scratch, params.device_index, params.host_bytes, tile_num);
    if (scratch_status != StagedMemoryCopyStatus::SUCCESS) {
        cleanup_local_scratch();
        return scratch_status;
    }

    const char* operation = "upload_device_ptrs";
    auto        err       = cudaMemcpyAsync(
        work_scratch->device_ptrs, h_ptrs.data(), tile_num * sizeof(void*), cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        operation = "upload_device_offsets";
        err       = cudaMemcpyAsync(
            work_scratch->device_offsets, h_offsets.data(), tile_num * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    }
    if (err == cudaSuccess) {
        operation = "upload_device_sizes";
        err       = cudaMemcpyAsync(
            work_scratch->device_sizes, h_sizes.data(), tile_num * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    }

    if (err == cudaSuccess && params.direction == StagedMemoryCopyDirection::H2D) {
        copyHostToPinnedStaging(params, work_scratch->host_staging);
        operation = "copy_h2d_staging_payload";
        err       = cudaMemcpyAsync(work_scratch->device_staging,
                              work_scratch->host_staging,
                              params.host_bytes,
                              cudaMemcpyHostToDevice,
                              stream);
        if (err == cudaSuccess) {
            operation = "scatter_kernel_launch";
            sDevMPS::launch_dsv4_memory_cache_scatter_copy_var_nooffset(
                work_scratch->device_staging,
                reinterpret_cast<const size_t*>(work_scratch->device_offsets),
                reinterpret_cast<const size_t*>(work_scratch->device_sizes),
                reinterpret_cast<void**>(work_scratch->device_ptrs),
                static_cast<int>(tile_num),
                0,
                stream);
            err = cudaGetLastError();
        }
    } else if (err == cudaSuccess) {
        operation = "gather_kernel_launch";
        sDevMPS::launch_dsv4_memory_cache_gather_copy_var_nooffset(
            reinterpret_cast<const void**>(work_scratch->device_ptrs),
            reinterpret_cast<const size_t*>(work_scratch->device_sizes),
            reinterpret_cast<const size_t*>(work_scratch->device_offsets),
            work_scratch->device_staging,
            static_cast<int>(tile_num),
            0,
            stream);
        err = cudaGetLastError();
        if (err == cudaSuccess) {
            operation = "copy_d2h_staging_payload";
            err       = cudaMemcpyAsync(work_scratch->host_staging,
                                  work_scratch->device_staging,
                                  params.host_bytes,
                                  cudaMemcpyDeviceToHost,
                                  stream);
        }
    }

    cudaError_t drain_error = cudaSuccess;
    if (err == cudaSuccess) {
        operation = "cudaStreamSynchronize";
        err       = cudaStreamSynchronize(stream);
    } else if (!isFatalCudaRuntimeError(static_cast<int>(err))) {
        // Drain earlier submissions only while the context is not known fatal.
        drain_error = cudaStreamSynchronize(stream);
    }
    if (err == cudaSuccess && params.direction == StagedMemoryCopyDirection::D2H) {
        copyPinnedStagingToHost(params, work_scratch->host_staging);
    }
    if (err != cudaSuccess) {
        // Preserve actual gather/scatter endpoints rather than claiming every
        // tile is a destination with a null source. Staging offsets are validated above.
        std::string context =
            "{\"host_staging\":" + std::to_string(reinterpret_cast<uintptr_t>(work_scratch->host_staging));
        context += ",\"host_capacity\":" + std::to_string(work_scratch->host_capacity);
        context += ",\"device_staging\":" + std::to_string(reinterpret_cast<uintptr_t>(work_scratch->device_staging));
        context += ",\"device_capacity\":" + std::to_string(work_scratch->device_capacity);
        context += ",\"device_ptrs\":" + std::to_string(reinterpret_cast<uintptr_t>(work_scratch->device_ptrs));
        context += ",\"device_offsets\":" + std::to_string(reinterpret_cast<uintptr_t>(work_scratch->device_offsets));
        context += ",\"device_sizes\":" + std::to_string(reinterpret_cast<uintptr_t>(work_scratch->device_sizes));
        context += ",\"metadata_capacity\":" + std::to_string(work_scratch->meta_capacity);
        context += ",\"drain_error\":" + std::to_string(static_cast<int>(drain_error));
        context += ",\"payload_bytes\":" + std::to_string(params.host_bytes);
        context += ",\"direction\":\"" + std::string(params.direction == StagedMemoryCopyDirection::H2D ? "H2D" : "D2H")
                   + "\"";
        context += ",\"host_segments\":[";
        for (size_t i = 0; i < params.host_segments.size(); ++i) {
            const auto& segment = params.host_segments[i];
            if (i) {
                context += ',';
            }
            context += "{\"host\":" + std::to_string(reinterpret_cast<uintptr_t>(segment.host)) + ",\"offset\":"
                       + std::to_string(segment.host_offset) + ",\"bytes\":" + std::to_string(segment.bytes) + "}";
        }
        context += "],\"staging_offsets\":[";
        for (size_t i = 0; i < h_offsets.size(); ++i) {
            if (i) {
                context += ',';
            }
            context += std::to_string(h_offsets[i]);
        }
        context += "]}";
        std::vector<void*>       error_dsts;
        std::vector<const void*> error_srcs;
        error_dsts.reserve(h_ptrs.size());
        error_srcs.reserve(h_ptrs.size());
        for (size_t i = 0; i < h_ptrs.size(); ++i) {
            void* staging = static_cast<uint8_t*>(work_scratch->device_staging) + h_offsets[i];
            error_dsts.push_back(params.direction == StagedMemoryCopyDirection::H2D ? h_ptrs[i] : staging);
            error_srcs.push_back(params.direction == StagedMemoryCopyDirection::H2D ? staging : h_ptrs[i]);
        }
        RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: tiles=%zu bytes=%zu direction=%s error=%s",
                            tile_num,
                            params.host_bytes,
                            params.direction == StagedMemoryCopyDirection::H2D ? "H2D" : "D2H",
                            cudaGetErrorString(err));
        recordFatalCopyError(err,
                             FatalCudaErrorSite::StagedCopy,
                             __FILE__,
                             __LINE__,
                             params.device_index,
                             stream,
                             error_dsts.data(),
                             error_srcs.data(),
                             h_sizes.data(),
                             h_ptrs.size(),
                             operation,
                             context);
        if (drain_error != cudaSuccess) {
            recordFatalCopyError(drain_error,
                                 FatalCudaErrorSite::StagedCopy,
                                 __FILE__,
                                 __LINE__,
                                 params.device_index,
                                 stream,
                                 error_dsts.data(),
                                 error_srcs.data(),
                                 h_sizes.data(),
                                 h_ptrs.size(),
                                 "drain_after_staged_failure",
                                 context);
        }
        cleanup_local_scratch();
        return StagedMemoryCopyStatus::EXECUTION_FAILED;
    }
    cleanup_local_scratch();
    check_cuda_error();
    return StagedMemoryCopyStatus::SUCCESS;
}

void warmupNoBlockCopy() {
    installCudacoreTerminateGuard();
    prepareCudacoreFlightRecorder();
    if (!warmupSplitKvCopyKernels(at::cuda::getCurrentCUDAStream().stream())) {
        RTP_LLM_LOG_WARNING("warmupSplitKvCopyKernels failed; split-KV copy may JIT on first use");
    }
}

}  // namespace rtp_llm
