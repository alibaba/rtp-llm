// Test-only staged-copy baseline from main e73826e27a6005640bb75ce2a73f235bfa51c55b.
// Bodies below are copied unchanged from NoBlockCopy.cc and
// DeviceHostCopyStrategy.cc; the namespace isolates their scratch and symbols.
#include "rtp_llm/models_py/bindings/cuda/test/MainStagedCopyBaseline.h"
#include "rtp_llm/models_py/bindings/cuda/test/MainStagedCopyKernel.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <algorithm>
#include <cstring>
#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm::crc_copy_benchmark::main_baseline {
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

static constexpr size_t kStagedAlignment = 16;

static size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

StagedSmDeviceHostCopyStrategy::~StagedSmDeviceHostCopyStrategy() {
    for (auto& [_, scratch] : scratch_by_device_) {
        if (scratch) {
            releaseStagedMemoryCopyScratch(*scratch);
        }
    }
}

StrategyResult StagedSmDeviceHostCopyStrategy::tryExecute(const DeviceHostCopyPlan&    plan,
                                                          const DeviceHostCopyOptions& options) {
    if (!options.staged_sm_copy_enabled) {
        return StrategyResult::notApplicable();
    }

    if (plan.copy_tiles.size() < options.staged_sm_min_tile_count) {
        return StrategyResult::notApplicable();
    }

    const int device_index = plan.copy_tiles.front().device_index;
    if (device_index < 0) {
        return StrategyResult::notApplicable();
    }

    size_t total_bytes = 0;
    for (const auto& tile : plan.copy_tiles) {
        total_bytes += tile.bytes;
    }

    if (total_bytes < options.staged_sm_min_bytes) {
        return StrategyResult::notApplicable();
    }

    // Build staged params with compact host segments
    StagedMemoryCopyParams staged_params;
    staged_params.host_base    = plan.host.base;
    staged_params.device_index = device_index;
    staged_params.direction    = plan.device_to_host ? StagedMemoryCopyDirection::D2H : StagedMemoryCopyDirection::H2D;

    size_t current_staging_offset = 0;
    staged_params.tiles.reserve(plan.copy_tiles.size());
    staged_params.host_segments.reserve(plan.copy_tiles.size());

    for (const auto& tile : plan.copy_tiles) {
        size_t staging_offset = alignUp(current_staging_offset, kStagedAlignment);

        StagedMemoryCopyTile staged_tile;
        staged_tile.gpu         = tile.device_addr;
        staged_tile.host_offset = staging_offset;
        staged_tile.bytes       = tile.bytes;
        staged_params.tiles.push_back(staged_tile);

        StagedMemoryCopyHostSegment segment;
        segment.host        = tile.host_addr;
        segment.host_offset = staging_offset;
        segment.bytes       = tile.bytes;

        // Merge with previous segment if contiguous in both host and staging space
        if (!staged_params.host_segments.empty()) {
            auto& prev     = staged_params.host_segments.back();
            auto* prev_end = static_cast<uint8_t*>(prev.host) + prev.bytes;
            if (prev_end == tile.host_addr && prev.host_offset + prev.bytes == staging_offset) {
                prev.bytes += tile.bytes;
                current_staging_offset = staging_offset + tile.bytes;
                continue;
            }
        }
        staged_params.host_segments.push_back(segment);
        current_staging_offset = staging_offset + tile.bytes;
    }
    staged_params.host_bytes = current_staging_offset;

    std::lock_guard<std::mutex> lock(scratch_mutex_);
    auto&                       entry = scratch_by_device_[device_index];
    if (!entry) {
        entry               = std::make_unique<StagedMemoryCopyScratch>();
        entry->device_index = device_index;
    }

    bool ok = execStagedMemoryCopy(staged_params, entry.get());
    if (!ok) {
        // Conservatively fall back to generic
        return StrategyResult::notApplicable();
    }
    return StrategyResult::done();
}

}  // namespace rtp_llm::crc_copy_benchmark::main_baseline
