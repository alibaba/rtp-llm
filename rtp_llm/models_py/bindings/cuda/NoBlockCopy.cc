#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/common/kernels/sm_copy_kernel.h"
#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace rtp_llm {

namespace {

enum class HostCoverage {
    Invalid,
    Partial,
    Full,
};

struct CopyStreamRegistry {
    std::mutex                                                    mutex;
    std::unordered_map<int, std::weak_ptr<DeviceHostCopyStreams>> entries;
};

CopyStreamRegistry& copyStreamRegistry() {
    static CopyStreamRegistry registry;
    return registry;
}

[[noreturn]] void
failCopyStreams(const DeviceHostCopyExecutionContext& context, const char* phase, size_t tile_num, cudaError_t error) {
    RTP_LLM_FAIL("device-host copy completion unconfirmed phase=%s device=%d direction=%s stream=%p tiles=%zu "
                 "error_code=%d error=%s",
                 phase,
                 context.deviceIndex(),
                 context.direction == DeviceHostCopyDirection::H2D ? "H2D" : "D2H",
                 reinterpret_cast<void*>(context.stream()),
                 tile_num,
                 static_cast<int>(error),
                 cudaGetErrorString(error));
}

class CudaDeviceHostCopyStreams final: public DeviceHostCopyStreams {
public:
    CudaDeviceHostCopyStreams(int device, cudaStream_t h2d, cudaStream_t d2h):
        DeviceHostCopyStreams(device, reinterpret_cast<uintptr_t>(h2d), reinterpret_cast<uintptr_t>(d2h)) {}

    ~CudaDeviceHostCopyStreams() override {
        try {
            c10::cuda::CUDAGuard guard(device_index);
            for (auto handle : {h2d_stream, d2h_stream}) {
                const auto error = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(handle));
                if (error != cudaSuccess) {
                    RTP_LLM_LOG_WARNING("copy stream destroy failed device=%d stream=%p: %s",
                                        device_index,
                                        reinterpret_cast<void*>(handle),
                                        cudaGetErrorString(error));
                }
            }
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING("copy stream teardown failed device=%d: %s", device_index, error.what());
        }
    }
};

bool validCopyContext(const DeviceHostCopyExecutionContext& context,
                      int                                   device_index,
                      DeviceHostCopyDirection               direction) {
    return context.owner && context.deviceIndex() == device_index && context.direction == direction
           && context.stream() != 0;
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

bool ensureStagedMemoryCopyScratch(StagedMemoryCopyScratch& scratch,
                                   int                      device_index,
                                   size_t                   host_bytes,
                                   size_t                   tile_num) {
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
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed to allocate pinned host staging: %s",
                                cudaGetErrorString(err));
            return false;
        }
        scratch.host_capacity = host_bytes;
    }

    if (scratch.device_capacity < host_bytes) {
        releaseDevicePointer(scratch.device_staging);
        auto err = cudaMalloc(&scratch.device_staging, host_bytes);
        if (err != cudaSuccess) {
            scratch.device_capacity = 0;
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed to allocate device staging: %s", cudaGetErrorString(err));
            return false;
        }
        scratch.device_capacity = host_bytes;
    }

    if (scratch.meta_capacity < tile_num) {
        releaseMetadataScratch(scratch);
        auto err = cudaMalloc(&scratch.device_ptrs, tile_num * sizeof(void*));
        if (err == cudaSuccess) {
            err = cudaMalloc(&scratch.device_offsets, tile_num * sizeof(size_t));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc(&scratch.device_sizes, tile_num * sizeof(size_t));
        }
        if (err != cudaSuccess) {
            releaseMetadataScratch(scratch);
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed to allocate device metadata: %s", cudaGetErrorString(err));
            return false;
        }
        scratch.meta_capacity = tile_num;
    }
    return true;
}

}  // namespace

std::shared_ptr<DeviceHostCopyStreams> acquireDeviceHostCopyStreams(int device_index) {
    if (device_index < 0) {
        throw std::invalid_argument("invalid copy stream device");
    }
    auto&                       registry = copyStreamRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto&                       entry = registry.entries[device_index];
    if (auto existing = entry.lock()) {
        return existing;
    }
    c10::cuda::CUDAGuard guard(device_index);
    cudaStream_t         h2d = nullptr;
    cudaStream_t         d2h = nullptr;
    check_cuda_value(cudaStreamCreateWithFlags(&h2d, cudaStreamNonBlocking));
    try {
        check_cuda_value(cudaStreamCreateWithFlags(&d2h, cudaStreamNonBlocking));
        if (h2d == d2h || h2d == nullptr || d2h == nullptr) {
            throw std::runtime_error("copy direction streams are not distinct non-default streams");
        }
        auto pair  = std::make_shared<CudaDeviceHostCopyStreams>(device_index, h2d, d2h);
        h2d        = nullptr;
        d2h        = nullptr;
        entry      = pair;
        RTP_LLM_LOG_INFO("copy streams initialized device=%d owner=%p h2d=%p d2h=%p",
                         device_index,
                         pair.get(),
                         reinterpret_cast<void*>(pair->h2d_stream),
                         reinterpret_cast<void*>(pair->d2h_stream));
        return pair;
    } catch (...) {
        if (d2h != nullptr) {
            cudaStreamDestroy(d2h);
        }
        if (h2d != nullptr) {
            cudaStreamDestroy(h2d);
        }
        throw;
    }
}

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

static void execNoBlockCopyImpl(const MultiCopyParams& params, const DeviceHostCopyExecutionContext* context) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    const bool has_cuda_tensor =
        !params.multi_dst.empty() && (params.multi_dst[0].is_cuda() || params.multi_src[0].is_cuda());
    const int copy_device =
        params.multi_dst.empty() ? getCopyDevice(-1, -1) : getCopyDevice(params.multi_dst[0], params.multi_src[0]);
    c10::cuda::CUDAGuard device_guard(copy_device);

    if (context != nullptr) {
        const auto direction = !params.multi_dst.empty() && params.multi_dst[0].is_cuda() ?
                                   DeviceHostCopyDirection::H2D :
                                   DeviceHostCopyDirection::D2H;
        if (!validCopyContext(*context, copy_device, direction) || params.split_kv_layer_num > 0) {
            throw std::invalid_argument("invalid explicit generic copy context");
        }
    }
    auto stream =
        context ? reinterpret_cast<cudaStream_t>(context->stream()) : getNoBlockCopyStream(copy_device).stream();

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

    bool submitted = false;
    try {
        for (size_t i = 0; i < params.multi_src.size(); ++i) {
            submitted = true;
            check_cuda_value(cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                                             params.multi_src[i].data_ptr(),
                                             params.multi_src[i].nbytes(),
                                             cudaMemcpyDefault,
                                             stream));
        }
        check_cuda_value(cudaStreamSynchronize(stream));
        check_cuda_error();
    } catch (...) {
        if (submitted) {
            const auto drain_error = cudaStreamSynchronize(stream);
            if (drain_error != cudaSuccess && context != nullptr) {
                failCopyStreams(*context, "generic_exception_drain", params.multi_src.size(), drain_error);
            }
        }
        throw;
    }
}

void execNoBlockCopy(const MultiCopyParams& params) {
    execNoBlockCopyImpl(params, nullptr);
}

void execNoBlockCopy(const MultiCopyParams& params, const DeviceHostCopyExecutionContext& context) {
    execNoBlockCopyImpl(params, &context);
}

static bool validCopyRegion(const MemoryCopyRegion& region) {
    if (region.width == 0) {
        return true;
    }
    if (region.src == nullptr || region.dst == nullptr || region.height == 0
        || region.src_pitch < region.width || region.dst_pitch < region.width) {
        return false;
    }
    const auto fits = [&](const void* ptr, size_t pitch) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
        return region.width <= UINTPTR_MAX - address
               && region.height - 1 <= (UINTPTR_MAX - address - region.width) / pitch;
    };
    return fits(region.src, region.src_pitch) && fits(region.dst, region.dst_pitch);
}

static BatchedMemoryCopyStatus execBatchedMemoryCopyImpl(
    const BatchedMemoryCopyParams& params, const DeviceHostCopyExecutionContext& context,
    const std::vector<MemoryCopyRegion>* regions = nullptr) {
    if (!validCopyContext(context, params.device_index, params.direction)) {
        return BatchedMemoryCopyStatus::EXECUTION_FAILED;
    }
    const bool use_3d = regions != nullptr;
    if (use_3d && !std::all_of(regions->begin(), regions->end(), validCopyRegion)) {
        RTP_LLM_LOG_WARNING("invalid CUDA 3D copy region");
        return BatchedMemoryCopyStatus::EXECUTION_FAILED;
    }
    if (use_3d ? regions->empty() : params.tiles.empty()) {
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
                ">=%d for the CUDA batch copy ABI, installed driver API version=%d; falling back to generic "
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
                            "runtime version (%s); cannot prove CUDA batch copy ABI compatibility",
                            CUDART_VERSION,
                            cudaGetErrorString(runtime_version_error));
        return BatchedMemoryCopyStatus::NOT_SUPPORTED;
    }
    if (runtime_version < 12080) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d, runtime version=%d "
                            "predates CUDA batch copy; falling back to generic copy",
                            CUDART_VERSION,
                            runtime_version);
        return BatchedMemoryCopyStatus::NOT_SUPPORTED;
    }
    const bool compiled_with_cuda13_batch_abi = CUDART_VERSION >= 13000;
    const bool runtime_uses_cuda13_batch_abi  = runtime_version >= 13000;
    if (compiled_with_cuda13_batch_abi != runtime_uses_cuda13_batch_abi) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d and runtime version=%d "
                            "use incompatible CUDA batch copy signatures; falling back to generic copy",
                            CUDART_VERSION,
                            runtime_version);
        return BatchedMemoryCopyStatus::NOT_SUPPORTED;
    }

    c10::cuda::CUDAGuard device_guard(params.device_index);
    auto                 stream = reinterpret_cast<cudaStream_t>(context.stream());

    const size_t             tile_num = use_3d ? regions->size() : params.tiles.size();
    std::vector<void*>       dsts;
    std::vector<const void*> srcs;
    std::vector<size_t>      sizes;
    std::vector<cudaMemcpy3DBatchOp> ops;
    if (use_3d) {
        ops.reserve(tile_num);
    } else {
        dsts.reserve(tile_num);
        srcs.reserve(tile_num);
        sizes.reserve(tile_num);
    }
    if (use_3d) {
        for (const auto& region : *regions) {
            if (region.width == 0) {
                continue;
            }
            cudaMemcpy3DBatchOp op{};
            op.src.type               = cudaMemcpyOperandTypePointer;
            op.src.op.ptr.ptr         = const_cast<void*>(region.src);
            op.src.op.ptr.rowLength   = region.src_pitch;
            op.src.op.ptr.layerHeight = region.height;
            op.dst.type               = cudaMemcpyOperandTypePointer;
            op.dst.op.ptr.ptr         = region.dst;
            op.dst.op.ptr.rowLength   = region.dst_pitch;
            op.dst.op.ptr.layerHeight = region.height;
            op.extent                 = {region.width, region.height, 1};
            op.srcAccessOrder         = cudaMemcpySrcAccessOrderStream;
            ops.push_back(op);
        }
    } else {
        for (const auto& tile : params.tiles) {
            if (tile.dst == nullptr || tile.src == nullptr || tile.bytes == 0) {
                continue;
            }
            dsts.push_back(tile.dst);
            srcs.push_back(tile.src);
            sizes.push_back(tile.bytes);
        }
    }
    const size_t copy_count = use_3d ? ops.size() : dsts.size();
    if (copy_count == 0) {
        return BatchedMemoryCopyStatus::SUCCESS;
    }

    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy api=%s phase=submit device=%d stream=%p tiles=%zu",
                      use_3d ? "cuda_3d_batch" : "cuda_batch",
                      params.device_index,
                      static_cast<void*>(stream),
                      copy_count);

    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx     = 0;
#if CUDART_VERSION < 13000
    std::vector<void*> mutable_srcs;
    mutable_srcs.reserve(srcs.size());
    for (auto* src : srcs) {
        mutable_srcs.push_back(const_cast<void*>(src));
    }
    size_t fail_idx = 0;
#endif
    cudaError_t submit_error;
#if CUDART_VERSION >= 13000
    if (use_3d) {
        submit_error = cudaMemcpy3DBatchAsync(ops.size(), ops.data(), 0, stream);
    } else {
        submit_error = cudaMemcpyBatchAsync(
            dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, stream);
    }
#else
    if (use_3d) {
        submit_error = cudaMemcpy3DBatchAsync(ops.size(), ops.data(), &fail_idx, 0, stream);
    } else {
        submit_error = cudaMemcpyBatchAsync(
            dsts.data(), mutable_srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, &fail_idx, stream);
    }
#endif
    if (submit_error != cudaSuccess) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy failed phase=submit device=%d stream=%p tiles=%zu error_code=%d "
                            "error=%s",
                            params.device_index,
                            static_cast<void*>(stream),
                            copy_count,
                            static_cast<int>(submit_error),
                            cudaGetErrorString(submit_error));
        const auto drain_error = cudaStreamSynchronize(stream);
        if (drain_error != cudaSuccess) {
            failCopyStreams(context, "batch_submit_drain", copy_count, drain_error);
        }
        return BatchedMemoryCopyStatus::EXECUTION_FAILED;
    }

    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy phase=submitted device=%d stream=%p tiles=%zu",
                      params.device_index,
                      static_cast<void*>(stream),
                      copy_count);

    const auto completion_error = cudaStreamSynchronize(stream);
    if (completion_error != cudaSuccess) {
        failCopyStreams(context, "batch_completion", copy_count, completion_error);
    }
    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy phase=completed device=%d stream=%p tiles=%zu",
                      params.device_index,
                      static_cast<void*>(stream),
                      copy_count);
    // cudaStreamSynchronize already reports deferred errors from this batch.
    // Do not call check_cuda_error() here: in DEBUG mode it performs a
    // device-wide synchronize and can wait on unrelated TP/NCCL work.
    if (Logger::getEngineLogger().isDebugMode()) {
        check_cuda_value(cudaGetLastError());
    }
    return BatchedMemoryCopyStatus::SUCCESS;
#else
    (void)use_3d;
    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy unavailable: CUDART_VERSION=%d", CUDART_VERSION);
    return BatchedMemoryCopyStatus::NOT_SUPPORTED;
#endif
}

BatchedMemoryCopyStatus execBatchedMemoryCopy(const BatchedMemoryCopyParams& params,
                                              const DeviceHostCopyExecutionContext& context) {
    return execBatchedMemoryCopyImpl(params, context);
}

BatchedMemoryCopyStatus execBatched3DMemoryCopy(const BatchedMemoryCopyParams& params,
                                                const DeviceHostCopyExecutionContext& context) {
    if (!validCopyContext(context, params.device_index, params.direction)) {
        return BatchedMemoryCopyStatus::EXECUTION_FAILED;
    }
    Batched3DMemoryCopyParams unmerged;
    unmerged.device_index = params.device_index;
    unmerged.regions.reserve(params.tiles.size());
    for (const auto& tile : params.tiles) {
        unmerged.regions.push_back({tile.src, tile.dst, tile.bytes, 1, tile.bytes, tile.bytes});
    }
    return execBatched3DMemoryCopy(unmerged, context);
}

BatchedMemoryCopyStatus execBatched3DMemoryCopy(const Batched3DMemoryCopyParams& params,
                                                const DeviceHostCopyExecutionContext& context) {
    BatchedMemoryCopyParams common;
    common.device_index = params.device_index;
    common.direction = context.direction;
    return execBatchedMemoryCopyImpl(common, context, &params.regions);
}

bool execStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch* scratch) {
    if (params.tiles.empty()) {
        return true;
    }
    if (params.device_index < 0 || params.host_bytes == 0 || !checkHostSegments(params)) {
        RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: device=%d host_base=%p host_bytes=%zu host_segments=%zu",
                            params.device_index,
                            params.host_base,
                            params.host_bytes,
                            params.host_segments.size());
        return false;
    }
    const auto host_coverage = checkHostCoverage(params);
    if (host_coverage == HostCoverage::Invalid) {
        RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: invalid/overlapping host coverage, tiles=%zu bytes=%zu",
                            params.tiles.size(),
                            params.host_bytes);
        return false;
    }

    std::vector<void*>  h_ptrs;
    std::vector<size_t> h_offsets;
    std::vector<size_t> h_sizes;
    h_ptrs.reserve(params.tiles.size());
    h_offsets.reserve(params.tiles.size());
    h_sizes.reserve(params.tiles.size());
    for (const auto& tile : params.tiles) {
        if (tile.gpu == nullptr || tile.bytes == 0) {
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: invalid tile gpu=%p bytes=%zu", tile.gpu, tile.bytes);
            return false;
        }
        if (tile.host_offset > params.host_bytes || tile.bytes > params.host_bytes - tile.host_offset) {
            RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: tile out of host span, off=%zu bytes=%zu host=%zu",
                                tile.host_offset,
                                tile.bytes,
                                params.host_bytes);
            return false;
        }
        h_ptrs.push_back(tile.gpu);
        h_offsets.push_back(tile.host_offset);
        h_sizes.push_back(tile.bytes);
    }
    if (h_ptrs.empty()) {
        return true;
    }

    check_cuda_value(cudaSetDevice(params.device_index));
    auto stream = getNoBlockCopyStream(params.device_index).stream();

    StagedMemoryCopyScratch local_scratch;
    auto*                   work_scratch          = scratch != nullptr ? scratch : &local_scratch;
    auto                    cleanup_local_scratch = [&]() {
        if (scratch == nullptr) {
            releaseStagedMemoryCopyScratch(local_scratch);
        }
    };

    const size_t tile_num = h_ptrs.size();
    if (!ensureStagedMemoryCopyScratch(*work_scratch, params.device_index, params.host_bytes, tile_num)) {
        cleanup_local_scratch();
        return false;
    }

    auto err = cudaMemcpyAsync(
        work_scratch->device_ptrs, h_ptrs.data(), tile_num * sizeof(void*), cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(
            work_scratch->device_offsets, h_offsets.data(), tile_num * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(
            work_scratch->device_sizes, h_sizes.data(), tile_num * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    }

    if (err == cudaSuccess && params.direction == StagedMemoryCopyDirection::H2D) {
        copyHostToPinnedStaging(params, work_scratch->host_staging);
        err = cudaMemcpyAsync(work_scratch->device_staging,
                              work_scratch->host_staging,
                              params.host_bytes,
                              cudaMemcpyHostToDevice,
                              stream);
        if (err == cudaSuccess) {
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
            err = cudaMemcpyAsync(work_scratch->host_staging,
                                  work_scratch->device_staging,
                                  params.host_bytes,
                                  cudaMemcpyDeviceToHost,
                                  stream);
        }
    }

    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    } else {
        (void)cudaStreamSynchronize(stream);
    }
    if (err == cudaSuccess && params.direction == StagedMemoryCopyDirection::D2H) {
        copyPinnedStagingToHost(params, work_scratch->host_staging);
    }
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: tiles=%zu bytes=%zu direction=%s error=%s",
                            tile_num,
                            params.host_bytes,
                            params.direction == StagedMemoryCopyDirection::H2D ? "H2D" : "D2H",
                            cudaGetErrorString(err));
        cleanup_local_scratch();
        return false;
    }
    cleanup_local_scratch();
    check_cuda_error();
    return true;
}

void warmupNoBlockCopy() {
    if (!warmupSplitKvCopyKernels(at::cuda::getCurrentCUDAStream().stream())) {
        RTP_LLM_LOG_WARNING("warmupSplitKvCopyKernels failed; split-KV copy may JIT on first use");
    }
}

}  // namespace rtp_llm
