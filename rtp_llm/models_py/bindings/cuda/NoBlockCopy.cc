#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/common/kernels/sm_copy_kernel.h"
#include "rtp_llm/models_py/bindings/cuda/SplitKvCacheCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <algorithm>
#include <cstring>
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

bool prewarmStagedMemoryCopyScratch(StagedMemoryCopyScratch& scratch,
                                    int                      device_index,
                                    size_t                   host_bytes,
                                    size_t                   tile_num) {
    if (host_bytes == 0 || tile_num == 0) {
        return false;
    }
    if (device_index < 0) {
        auto err = cudaGetDevice(&device_index);
        if (err != cudaSuccess) {
            RTP_LLM_LOG_WARNING("prewarmStagedMemoryCopyScratch failed to query current device: %s",
                                cudaGetErrorString(err));
            return false;
        }
    }
    if (!ensureStagedMemoryCopyScratch(scratch, device_index, host_bytes, tile_num)) {
        return false;
    }
    // First launch of the gather/scatter kernels lazy-loads their CUDA module, which
    // synchronizes the context; do it here while the device is idle instead of inside
    // execStagedMemoryCopy where it can deadlock against a spinning collective kernel.
    if (!sDevMPS::warmup_sm_copy_var_nooffset_kernels(getNoBlockCopyStream(device_index).stream())) {
        RTP_LLM_LOG_WARNING(
            "warmup_sm_copy_var_nooffset_kernels failed; staged copy may lazy-load kernels on first use");
    }
    return true;
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

bool execBatchedMemoryCopy(const BatchedMemoryCopyParams& params) {
    if (params.tiles.empty()) {
        return true;
    }
    if (params.device_index < 0) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy failed: invalid device_index=%d", params.device_index);
        return false;
    }

#if CUDART_VERSION >= 12080
    int        runtime_version       = 0;
    const auto runtime_version_error = cudaRuntimeGetVersion(&runtime_version);
    if (runtime_version_error != cudaSuccess) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d, failed to query "
                            "runtime version (%s); cannot prove cudaMemcpyBatchAsync ABI compatibility",
                            CUDART_VERSION,
                            cudaGetErrorString(runtime_version_error));
        return false;
    }
    if (runtime_version < 12080) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d, runtime version=%d "
                            "predates cudaMemcpyBatchAsync; falling back to generic copy",
                            CUDART_VERSION,
                            runtime_version);
        return false;
    }
    const bool compiled_with_cuda13_batch_abi = CUDART_VERSION >= 13000;
    const bool runtime_uses_cuda13_batch_abi  = runtime_version >= 13000;
    if (compiled_with_cuda13_batch_abi != runtime_uses_cuda13_batch_abi) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy unavailable: compile-time CUDART_VERSION=%d and runtime version=%d "
                            "use incompatible cudaMemcpyBatchAsync signatures; falling back to generic copy",
                            CUDART_VERSION,
                            runtime_version);
        return false;
    }

    check_cuda_value(cudaSetDevice(params.device_index));
    auto stream = getNoBlockCopyStream(params.device_index).stream();

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
        return true;
    }

    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx     = 0;
#if CUDART_VERSION >= 13000
    auto err = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, stream);
#else
    std::vector<void*> mutable_srcs;
    mutable_srcs.reserve(srcs.size());
    for (auto* src : srcs) {
        mutable_srcs.push_back(const_cast<void*>(src));
    }
    size_t fail_idx = 0;
    auto   err      = cudaMemcpyBatchAsync(
        dsts.data(), mutable_srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, &fail_idx, stream);
#endif
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING("execBatchedMemoryCopy failed: tiles=%zu, error=%s", dsts.size(), cudaGetErrorString(err));
        return false;
    }
    check_cuda_error();
    return true;
#else
    RTP_LLM_LOG_DEBUG("execBatchedMemoryCopy unavailable: CUDART_VERSION=%d", CUDART_VERSION);
    return false;
#endif
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
    auto stream = getNoBlockCopyStream(params.device_index).stream();

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

void warmupNoBlockCopy() {
    if (!warmupSplitKvCopyKernels(at::cuda::getCurrentCUDAStream().stream())) {
        RTP_LLM_LOG_WARNING("warmupSplitKvCopyKernels failed; split-KV copy may JIT on first use");
    }
}

}  // namespace rtp_llm
