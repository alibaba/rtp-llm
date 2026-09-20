#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/common/kernels/sm_copy_kernel.h"
#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/models_py/bindings/cuda/LinearCheckpointCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace rtp_llm {

namespace {

at::cuda::CUDAStream getNoBlockCopyStream() {
    int device = 0;
    check_cuda_value(cudaGetDevice(&device));
    return at::cuda::getStreamFromExternal(getCacheCopyStream(), device);
}

enum class HostCoverage {
    Invalid,
    Partial,
    Full,
};

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

size_t nextHostScratchCapacity(size_t current_capacity, size_t required_capacity) {
    constexpr size_t kMiB            = 1024ULL * 1024ULL;
    constexpr size_t kMaxGrowthBytes = 64ULL * kMiB;
    constexpr size_t kMinGrowthBytes = 1ULL * kMiB;
    constexpr size_t kCapacityAlign  = 1ULL * kMiB;

    if (required_capacity <= current_capacity) {
        return current_capacity;
    }

    size_t target_capacity = required_capacity;
    if (current_capacity == 0) {
        target_capacity = std::max(target_capacity, kMinGrowthBytes);
    } else {
        const size_t growth = std::clamp(current_capacity / 2, kMinGrowthBytes, kMaxGrowthBytes);
        if (current_capacity <= std::numeric_limits<size_t>::max() - growth) {
            target_capacity = std::max(target_capacity, current_capacity + growth);
        }
    }
    if (target_capacity > std::numeric_limits<size_t>::max() - (kCapacityAlign - 1)) {
        return required_capacity;
    }
    return (target_capacity + kCapacityAlign - 1) / kCapacityAlign * kCapacityAlign;
}

bool ensureStagedMemoryCopyScratch(
    StagedMemoryCopyScratch& scratch, int device_index, size_t host_bytes, bool need_host_staging, size_t tile_num) {
    if (scratch.device_index >= 0 && scratch.device_index != device_index) {
        releaseStagedMemoryCopyScratch(scratch);
    }
    check_cuda_value(cudaSetDevice(device_index));
    scratch.device_index = device_index;

    if (need_host_staging && scratch.host_capacity < host_bytes) {
        const size_t new_capacity = nextHostScratchCapacity(scratch.host_capacity, host_bytes);
        void*        new_staging  = nullptr;
        const auto   alloc_begin  = std::chrono::steady_clock::now();
        auto         err          = cudaHostAlloc(&new_staging, new_capacity, cudaHostAllocDefault);
        const auto   alloc_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - alloc_begin)
                .count();
        if (err != cudaSuccess) {
            RTP_LLM_LOG_WARNING(
                "execStagedMemoryCopy failed to grow pinned host staging, requested=%zu old_capacity=%zu "
                "target_capacity=%zu cost_us=%ld error=%s",
                host_bytes,
                scratch.host_capacity,
                new_capacity,
                alloc_us,
                cudaGetErrorString(err));
            return false;
        }
        void*        old_staging  = scratch.host_staging;
        const size_t old_capacity = scratch.host_capacity;
        scratch.host_staging      = new_staging;
        scratch.host_capacity     = new_capacity;
        ++scratch.host_allocation_count;
        if (old_staging != nullptr) {
            const auto free_err = cudaFreeHost(old_staging);
            if (free_err != cudaSuccess) {
                RTP_LLM_LOG_WARNING(
                    "execStagedMemoryCopy failed to release old pinned host staging, old_capacity=%zu error=%s",
                    old_capacity,
                    cudaGetErrorString(free_err));
            }
        }
        RTP_LLM_LOG_INFO(
            "execStagedMemoryCopy grew pinned host staging, requested=%zu old_capacity=%zu new_capacity=%zu "
            "allocation_count=%zu cost_us=%ld",
            host_bytes,
            old_capacity,
            new_capacity,
            scratch.host_allocation_count,
            alloc_us);
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

cudaError_t
copyPinnedHostSegmentsToDeviceStaging(const StagedMemoryCopyParams& params, void* device_staging, cudaStream_t stream) {
    if (params.host_segments.empty()) {
        return cudaSuccess;
    }
#if CUDART_VERSION >= 12080
    std::vector<void*>       dsts;
    std::vector<const void*> srcs;
    std::vector<size_t>      sizes;
    dsts.reserve(params.host_segments.size());
    srcs.reserve(params.host_segments.size());
    sizes.reserve(params.host_segments.size());
    auto* device_base = static_cast<char*>(device_staging);
    for (const auto& segment : params.host_segments) {
        dsts.push_back(device_base + segment.host_offset);
        srcs.push_back(segment.host);
        sizes.push_back(segment.bytes);
    }
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx     = 0;
#if CUDART_VERSION >= 13000
    return cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, stream);
#else
    std::vector<void*> mutable_srcs;
    mutable_srcs.reserve(srcs.size());
    for (auto* src : srcs) {
        mutable_srcs.push_back(const_cast<void*>(src));
    }
    size_t fail_idx = 0;
    return cudaMemcpyBatchAsync(
        dsts.data(), mutable_srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, &fail_idx, stream);
#endif
#else
    auto* device_base = static_cast<char*>(device_staging);
    for (const auto& segment : params.host_segments) {
        const auto err = cudaMemcpyAsync(
            device_base + segment.host_offset, segment.host, segment.bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return err;
        }
    }
    return cudaSuccess;
#endif
}

cudaError_t submitBatchedMemoryCopy(const BatchedMemoryCopyParams& params, cudaStream_t stream) {
#if CUDART_VERSION >= 12080
    const size_t             tile_num = params.tiles.size();
    std::vector<void*>       dsts;
    std::vector<const void*> srcs;
    std::vector<size_t>      sizes;
    dsts.reserve(tile_num);
    srcs.reserve(tile_num);
    sizes.reserve(tile_num);
    for (const auto& tile : params.tiles) {
        if (tile.dst == nullptr || tile.src == nullptr || tile.bytes == 0) {
            continue;
        }
        dsts.push_back(tile.dst);
        srcs.push_back(tile.src);
        sizes.push_back(tile.bytes);
    }
    if (dsts.empty()) {
        return cudaSuccess;
    }

    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx     = 0;
#if CUDART_VERSION >= 13000
    return cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, stream);
#else
    std::vector<void*> mutable_srcs;
    mutable_srcs.reserve(srcs.size());
    for (auto* src : srcs) {
        mutable_srcs.push_back(const_cast<void*>(src));
    }
    size_t fail_idx = 0;
    return cudaMemcpyBatchAsync(
        dsts.data(), mutable_srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, &fail_idx, stream);
#endif
#else
    // Older CUDA runtimes retain the same stream/lifetime contract.
    for (const auto& tile : params.tiles) {
        if (tile.dst && tile.src && tile.bytes) {
            const auto err = cudaMemcpyAsync(tile.dst, tile.src, tile.bytes, cudaMemcpyDefault, stream);
            if (err != cudaSuccess) {
                return err;
            }
        }
    }
    return cudaSuccess;
#endif
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
    scratch.host_staging          = nullptr;
    scratch.host_capacity         = 0;
    scratch.host_allocation_count = 0;
    scratch.device_capacity       = 0;
    scratch.device_index          = -1;
}

void execNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    int copy_device = -1;
    if (!params.multi_dst.empty()) {
        if (params.multi_dst[0].is_cuda()) {
            copy_device = static_cast<int>(params.multi_dst[0].get_device());
        } else if (params.multi_src[0].is_cuda()) {
            copy_device = static_cast<int>(params.multi_src[0].get_device());
        }
        if (copy_device >= 0) {
            check_cuda_value(cudaSetDevice(copy_device));
        }
    }

    auto stream = getNoBlockCopyStream().stream();

    if (params.split_kv_layer_num > 0 && copy_device >= 0) {
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

bool execBatchedMemoryCopy(const BatchedMemoryCopyParams& params) {
    if (params.tiles.empty()) {
        return true;
    }
    if (params.device_index < 0) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy failed: invalid device_index=%d", params.device_index);
        return false;
    }

#if CUDART_VERSION >= 12080
    check_cuda_value(cudaSetDevice(params.device_index));
    auto stream = getNoBlockCopyStream().stream();

    auto err = submitBatchedMemoryCopy(params, stream);
    // A failed batch may already have submitted copies. Drain them before
    // callers release storage or attempt a fallback on the same destinations.
    const auto sync_error = cudaStreamSynchronize(stream);
    if (err == cudaSuccess) {
        err = sync_error;
    }
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING(
            "execBatchedMemoryCopy failed: tiles=%zu, error=%s", params.tiles.size(), cudaGetErrorString(err));
        return false;
    }
    check_cuda_error();
    return true;
#else
    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy unavailable: CUDART_VERSION=%d", CUDART_VERSION);
    return false;
#endif
}

bool execLinearCheckpointCopy(const BatchedMemoryCopyParams&               params,
                              const std::vector<LinearCheckpointCopyTile>& checkpoints,
                              bool                                         host_to_device) {
    if (checkpoints.empty()) {
        return execBatchedMemoryCopy(params);
    }
    if (params.device_index < 0 || checkpoints.size() > 65535) {
        return false;
    }
    const auto packedBytes = [](const LinearCheckpointCopyTile& tile) {
        return static_cast<size_t>(tile.heads) * tile.key_dim
               * (tile.dtype == LinearCheckpointDType::BF16 ? tile.value_dim * sizeof(uint16_t) :
                                                              tile.value_dim * sizeof(int8_t) + sizeof(float));
    };
    size_t bytes        = 0;
    int    max_channels = 0;
    for (const auto& tile : checkpoints) {
        if (!tile.state || !tile.host || tile.heads <= 0 || tile.value_dim <= 0 || tile.key_dim <= 0
            || (static_cast<size_t>(tile.heads) * tile.value_dim * tile.key_dim) % sizeof(float) != 0) {
            return false;
        }
        const int channels = tile.heads * tile.key_dim;
        max_channels       = std::max(max_channels, channels);
        bytes += packedBytes(tile);
    }
    c10::cuda::CUDAGuard       device_guard(params.device_index);
    auto                       stream = getNoBlockCopyStream();
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, params.device_index);
    auto packed  = torch::empty({static_cast<int64_t>(bytes)}, options);
    auto metadata =
        torch::empty({static_cast<int64_t>(checkpoints.size() * sizeof(LinearCheckpointDeviceTile))}, options);
    auto  host_metadata            = torch::empty_like(metadata, options.device(torch::kCPU).pinned_memory(true));
    auto* tiles                    = reinterpret_cast<LinearCheckpointDeviceTile*>(host_metadata.data_ptr());
    BatchedMemoryCopyParams copies = params;
    size_t                  offset = 0;
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        const auto& tile    = checkpoints[i];
        auto*       staging = static_cast<int8_t*>(packed.data_ptr()) + offset;
        tiles[i]            = {tile.state, staging, tile.heads, tile.value_dim, tile.key_dim, tile.dtype};
        const size_t size   = packedBytes(tile);
        copies.tiles.push_back(host_to_device ? BatchedMemoryCopyTile{staging, tile.host, size} :
                                                BatchedMemoryCopyTile{tile.host, staging, size});
        offset += size;
    }
    auto error = cudaMemcpyAsync(
        metadata.data_ptr(), host_metadata.data_ptr(), metadata.nbytes(), cudaMemcpyHostToDevice, stream.stream());
    auto* device_tiles = reinterpret_cast<const LinearCheckpointDeviceTile*>(metadata.data_ptr());
    if (error == cudaSuccess && !host_to_device) {
        invokeLinearCheckpointCopy(device_tiles, checkpoints.size(), max_channels, false, stream.stream());
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) {
        error = submitBatchedMemoryCopy(copies, stream.stream());
    }
    if (error == cudaSuccess && host_to_device) {
        invokeLinearCheckpointCopy(device_tiles, checkpoints.size(), max_channels, true, stream.stream());
        error = cudaGetLastError();
    }
    // Also drain on failure: staging and metadata must outlive submitted work.
    const auto sync_error = cudaStreamSynchronize(stream.stream());
    if (error == cudaSuccess) {
        error = sync_error;
    }
    if (error != cudaSuccess) {
        RTP_LLM_LOG_WARNING("Linear checkpoint copy failed: %s", cudaGetErrorString(error));
    }
    return error == cudaSuccess;
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
    if (params.direct_pinned_host_segments
        && (params.direction != StagedMemoryCopyDirection::H2D || params.host_segments.empty())) {
        RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: direct pinned segments require non-empty H2D segments");
        return false;
    }
    const auto host_coverage = checkHostCoverage(params);
    if (host_coverage == HostCoverage::Invalid) {
        RTP_LLM_LOG_WARNING("execStagedMemoryCopy failed: invalid/overlapping host coverage, tiles=%zu bytes=%zu",
                            params.tiles.size(),
                            params.host_bytes);
        return false;
    }

    check_cuda_value(cudaSetDevice(params.device_index));
    auto stream = getNoBlockCopyStream().stream();

    std::vector<void*>  h_ptrs;
    std::vector<size_t> h_offsets;
    std::vector<size_t> h_sizes;
    h_ptrs.reserve(params.tiles.size());
    h_offsets.reserve(params.tiles.size());
    h_sizes.reserve(params.tiles.size());
    for (const auto& tile : params.tiles) {
        if (tile.gpu == nullptr || tile.bytes == 0) {
            continue;
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

    StagedMemoryCopyScratch local_scratch;
    auto*                   work_scratch          = scratch != nullptr ? scratch : &local_scratch;
    auto                    cleanup_local_scratch = [&]() {
        if (scratch == nullptr) {
            releaseStagedMemoryCopyScratch(local_scratch);
        }
    };

    const size_t tile_num = h_ptrs.size();
    if (!ensureStagedMemoryCopyScratch(
            *work_scratch, params.device_index, params.host_bytes, !params.direct_pinned_host_segments, tile_num)) {
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
        if (params.direct_pinned_host_segments) {
            err = copyPinnedHostSegmentsToDeviceStaging(params, work_scratch->device_staging, stream);
        } else {
            copyHostToPinnedStaging(params, work_scratch->host_staging);
            err = cudaMemcpyAsync(work_scratch->device_staging,
                                  work_scratch->host_staging,
                                  params.host_bytes,
                                  cudaMemcpyHostToDevice,
                                  stream);
        }
        if (err == cudaSuccess) {
            sDevMPS::launch_dsv4_memory_cache_scatter_copy_var_nooffset(
                work_scratch->device_staging,
                reinterpret_cast<const size_t*>(work_scratch->device_offsets),
                reinterpret_cast<const size_t*>(work_scratch->device_sizes),
                reinterpret_cast<void**>(work_scratch->device_ptrs),
                static_cast<int>(tile_num),
                params.sm_copy_block_num,
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
            params.sm_copy_block_num,
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
