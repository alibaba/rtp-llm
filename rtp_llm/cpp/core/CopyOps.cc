#include "rtp_llm/cpp/core/CopyOps.h"
#include "rtp_llm/cpp/core/OpStatus.h"
#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <ATen/Dispatch.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <unistd.h>
#include <unordered_map>

#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/kernels/BatchCopyKernel.h"
#include "rtp_llm/cpp/kernels/FusedCopyKernel.h"
#include "rtp_llm/cpp/kernels/MultiCopyKernel.h"
#include "rtp_llm/cpp/runtime/DeviceError.h"
#endif

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "ATen/ops/cat.h"
#include "rtp_llm/cpp/core/SplitKvCacheCopy.h"
#include "rtp_llm/cpp/kernels/MaskLogitsKernel.h"
#include "rtp_llm/cpp/kernels/SplitKvCopyKernel.h"
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/runtime/rocm/CudaShims.h"
#endif

namespace rtp_llm {

namespace {

torch::Tensor contiguousCpuTensor(const torch::Tensor& tensor) {
    if (tensor.device().is_cpu() && tensor.is_contiguous()) {
        return tensor;
    }
    return tensor.cpu().contiguous();
}

void validatePackedMaskLogitsInputs(const torch::Tensor& logits,
                                    const torch::Tensor& packed_allow_mask,
                                    const torch::Tensor& row_indices,
                                    size_t               vocab_size) {
    RTP_LLM_CHECK_WITH_INFO(logits.defined() && (logits.dim() == 1 || logits.dim() == 2),
                            "packed mask logits must be a defined 1D or 2D tensor");
    const int64_t logits_columns = logits.dim() == 1 ? logits.size(0) : logits.size(1);
    const int64_t logits_rows    = logits.dim() == 1 ? 1 : logits.size(0);
    const int64_t logits_stride  = logits.dim() == 1 ? logits_columns : logits.stride(0);
    RTP_LLM_CHECK_WITH_INFO(logits.stride(logits.dim() - 1) == 1 && logits_stride >= logits_columns,
                            "packed mask logits rows must be non-overlapping and contiguous in the vocab dimension");
    RTP_LLM_CHECK_WITH_INFO(packed_allow_mask.defined() && packed_allow_mask.dim() == 2
                                && packed_allow_mask.scalar_type() == torch::kInt32 && packed_allow_mask.stride(1) == 1
                                && packed_allow_mask.stride(0) >= packed_allow_mask.size(1),
                            "packed allow mask rows must be non-overlapping 2D int32 data contiguous in the "
                            "bitmask dimension");
    if (row_indices.defined()) {
        RTP_LLM_CHECK_WITH_INFO(row_indices.dim() == 1 && row_indices.scalar_type() == torch::kInt32
                                    && row_indices.is_contiguous(),
                                "packed mask row indices must be a contiguous 1D int32 tensor");
        RTP_LLM_CHECK_WITH_INFO(packed_allow_mask.size(0) == row_indices.numel(),
                                "packed mask rows (%lld) must equal row index count (%lld)",
                                static_cast<long long>(packed_allow_mask.size(0)),
                                static_cast<long long>(row_indices.numel()));
    } else {
        RTP_LLM_CHECK_WITH_INFO(packed_allow_mask.size(0) <= logits_rows,
                                "identity-mapped packed mask rows (%lld) exceed logits rows (%lld)",
                                static_cast<long long>(packed_allow_mask.size(0)),
                                static_cast<long long>(logits_rows));
    }
    RTP_LLM_CHECK_WITH_INFO(vocab_size > 0 && vocab_size <= static_cast<size_t>(logits_columns),
                            "packed mask vocab_size=%zu must be in (0, logits columns=%lld]",
                            vocab_size,
                            static_cast<long long>(logits_columns));
    RTP_LLM_CHECK_WITH_INFO(logits_rows <= std::numeric_limits<int>::max()
                                && logits_stride <= std::numeric_limits<int>::max()
                                && packed_allow_mask.size(0) <= std::numeric_limits<int>::max()
                                && packed_allow_mask.size(1) <= std::numeric_limits<int>::max()
                                && packed_allow_mask.stride(0) <= std::numeric_limits<int>::max()
                                && vocab_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                            "packed mask tensor dimensions exceed kernel int32 limits");
    const size_t required_words = (vocab_size + 31) / 32;
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(packed_allow_mask.size(1)) >= required_words,
                            "packed mask width=%lld is smaller than required words=%zu for vocab_size=%zu",
                            static_cast<long long>(packed_allow_mask.size(1)),
                            required_words,
                            vocab_size);
}

#if USING_CUDA
template<typename Stream>
void launchPackedMaskLogits(const torch::Tensor& logits,
                            const torch::Tensor& packed_allow_mask,
                            const torch::Tensor& row_indices,
                            size_t               vocab_size,
                            Stream               stream) {
    validatePackedMaskLogitsInputs(logits, packed_allow_mask, row_indices, vocab_size);
    RTP_LLM_CHECK_WITH_INFO(logits.is_cuda(), "packed mask CUDA logits must be on a CUDA device");
    RTP_LLM_CHECK_WITH_INFO(logits.device() == packed_allow_mask.device(),
                            "packed mask CUDA logits and mask must be on the same device");
    if (row_indices.defined()) {
        RTP_LLM_CHECK_WITH_INFO(logits.device() == row_indices.device(),
                                "packed mask CUDA logits and row indices must be on the same device");
    }
    const int mask_rows   = static_cast<int>(packed_allow_mask.size(0));
    const int logits_rows = logits.dim() == 1 ? 1 : static_cast<int>(logits.size(0));
    const int logits_row_stride =
        logits.dim() == 1 ? static_cast<int>(logits.size(0)) : static_cast<int>(logits.stride(0));
    const int      bitmask_stride = static_cast<int>(packed_allow_mask.stride(0));
    const int      bitmask_words  = static_cast<int>(packed_allow_mask.size(1));
    const int32_t* row_index_data = row_indices.defined() ? row_indices.data_ptr<int32_t>() : nullptr;

    if (mask_rows == 0) {
        return;
    }

    if (logits.scalar_type() == torch::kFloat32) {
        invokePackedMaskLogits<float>(logits.data_ptr<float>(),
                                      packed_allow_mask.data_ptr<int32_t>(),
                                      row_index_data,
                                      mask_rows,
                                      logits_rows,
                                      logits_row_stride,
                                      static_cast<int>(vocab_size),
                                      bitmask_stride,
                                      bitmask_words,
                                      stream);
    } else if (logits.scalar_type() == torch::kFloat16) {
        invokePackedMaskLogits<half>(reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
                                     packed_allow_mask.data_ptr<int32_t>(),
                                     row_index_data,
                                     mask_rows,
                                     logits_rows,
                                     logits_row_stride,
                                     static_cast<int>(vocab_size),
                                     bitmask_stride,
                                     bitmask_words,
                                     stream);
    } else if (logits.scalar_type() == torch::kBFloat16) {
        invokePackedMaskLogits<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
                                              packed_allow_mask.data_ptr<int32_t>(),
                                              row_index_data,
                                              mask_rows,
                                              logits_rows,
                                              logits_row_stride,
                                              static_cast<int>(vocab_size),
                                              bitmask_stride,
                                              bitmask_words,
                                              stream);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}
#endif

void applyPackedMaskLogitsCpuFallback(const torch::Tensor& logits,
                                      const torch::Tensor& packed_allow_mask,
                                      const torch::Tensor& row_indices,
                                      size_t               vocab_size) {
    validatePackedMaskLogitsInputs(logits, packed_allow_mask, row_indices, vocab_size);

    // ROCm and other backends without a native packed-mask kernel deliberately
    // fall back to CPU. The work is O(mask_rows * vocab_size) over only the
    // current logits and mask; it never replays the generated prefix. Blocking
    // copies keep the temporary CPU tensors alive until masked logits reach the
    // caller's device.
    auto logits_cpu = contiguousCpuTensor(logits);
    auto mask_cpu   = contiguousCpuTensor(packed_allow_mask);
    auto rows_cpu   = row_indices.defined() ? contiguousCpuTensor(row_indices) : torch::Tensor{};

    const int64_t logits_rows    = logits_cpu.dim() == 1 ? 1 : logits_cpu.size(0);
    const int64_t logits_columns = logits_cpu.dim() == 1 ? logits_cpu.size(0) : logits_cpu.size(1);
    const int64_t mask_rows      = mask_cpu.size(0);
    const int64_t bitmask_words  = mask_cpu.size(1);
    const auto*   mask_data      = mask_cpu.data_ptr<int32_t>();
    const auto*   row_data       = rows_cpu.defined() ? rows_cpu.data_ptr<int32_t>() : nullptr;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, logits_cpu.scalar_type(), "packedMaskLogitsCpuFallback", [&] {
            auto* logits_data = logits_cpu.data_ptr<scalar_t>();
            for (int64_t compact_row = 0; compact_row < mask_rows; ++compact_row) {
                const int64_t logits_row = row_data == nullptr ? compact_row : row_data[compact_row];
                if (logits_row < 0 || logits_row >= logits_rows) {
                    continue;
                }
                const auto* mask_row        = mask_data + compact_row * bitmask_words;
                auto*       logits_row_data = logits_data + logits_row * logits_columns;
                for (size_t token = 0; token < vocab_size; ++token) {
                    const uint32_t word = static_cast<uint32_t>(mask_row[token / 32]);
                    if ((word & (1u << (token % 32))) == 0u) {
                        logits_row_data[token] = static_cast<scalar_t>(-std::numeric_limits<float>::max());
                    }
                }
            }
        });

    if (logits.data_ptr() != logits_cpu.data_ptr()) {
        logits.copy_(logits_cpu, /*non_blocking=*/false);
    }
}

}  // namespace

#if USING_CUDA
using DeviceGuard = c10::cuda::CUDAGuard;
#endif

namespace {
#if USING_CUDA
at::cuda::CUDAStream& getOverlapStream() {
    static thread_local auto s = at::cuda::getStreamFromPool(/*isHighPriority=*/true);
    return s;
}

at::cuda::CUDAStream getNoBlockCopyStream(int device_id) {
    static thread_local std::unordered_map<int, at::cuda::CUDAStream> streams;
    auto                                                              stream = streams.find(device_id);
    if (stream == streams.end()) {
        stream = streams.emplace(device_id, at::cuda::getStreamFromPool(/*isHighPriority=*/false, device_id)).first;
    }
    return stream->second;
}

int getCopyDevice(const torch::Tensor& dst, const torch::Tensor& src) {
    if (dst.is_cuda()) {
        return static_cast<int>(dst.get_device());
    }
    if (src.is_cuda()) {
        return static_cast<int>(src.get_device());
    }
    return static_cast<int>(at::cuda::current_device());
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
        has_gap = has_gap || offset > covered;
        covered = offset + bytes;
    }
    if (covered > params.host_bytes) {
        return HostCoverage::Invalid;
    }
    return !has_gap && covered == params.host_bytes ? HostCoverage::Full : HostCoverage::Partial;
}

bool checkHostSegments(const StagedMemoryCopyParams& params) {
    if (params.host_segments.empty()) {
        return params.host_base != nullptr && params.host_bytes > 0;
    }

    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(params.host_segments.size());
    for (const auto& segment : params.host_segments) {
        if (segment.host == nullptr || segment.bytes == 0 || segment.host_offset > params.host_bytes
            || segment.bytes > params.host_bytes - segment.host_offset) {
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

void copyHostToPinnedStaging(const StagedMemoryCopyParams& params, void* host_staging) {
    if (params.host_segments.empty()) {
        std::memcpy(host_staging, params.host_base, params.host_bytes);
        return;
    }
    auto* base = static_cast<char*>(host_staging);
    for (const auto& segment : params.host_segments) {
        std::memcpy(base + segment.host_offset, segment.host, segment.bytes);
    }
}

void copyPinnedStagingToHost(const StagedMemoryCopyParams& params, const void* host_staging) {
    if (params.host_segments.empty()) {
        std::memcpy(params.host_base, host_staging, params.host_bytes);
        return;
    }
    const auto* base = static_cast<const char*>(host_staging);
    for (const auto& segment : params.host_segments) {
        std::memcpy(segment.host, base + segment.host_offset, segment.bytes);
    }
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
    RTP_LLM_DEVICE_CHECK(cudaSetDevice(device_index));
    scratch.device_index = device_index;

    if (scratch.host_capacity < host_bytes) {
        if (scratch.host_staging != nullptr) {
            (void)cudaFreeHost(scratch.host_staging);
            scratch.host_staging = nullptr;
            scratch.host_capacity = 0;
        }
        const auto err = cudaHostAlloc(&scratch.host_staging, host_bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            RTP_LLM_LOG_WARNING("runtimeStagedMemoryCopy failed to allocate pinned host staging: %s",
                                cudaGetErrorString(err));
            return false;
        }
        scratch.host_capacity = host_bytes;
    }

    if (scratch.device_capacity < host_bytes) {
        releaseDevicePointer(scratch.device_staging);
        const auto err = cudaMalloc(&scratch.device_staging, host_bytes);
        if (err != cudaSuccess) {
            scratch.device_capacity = 0;
            RTP_LLM_LOG_WARNING("runtimeStagedMemoryCopy failed to allocate device staging: %s",
                                cudaGetErrorString(err));
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
            RTP_LLM_LOG_WARNING("runtimeStagedMemoryCopy failed to allocate device metadata: %s",
                                cudaGetErrorString(err));
            return false;
        }
        scratch.meta_capacity = tile_num;
    }
    return true;
}

#elif USING_ROCM
at::hip::HIPStream& getOverlapStream() {
    static thread_local auto s = at::hip::getStreamFromPool(/*isHighPriority=*/true);
    return s;
}
#endif
}  // anonymous namespace

#if USING_CUDA

// ============================================================
// Copy ops (CUDA)
// ============================================================

void runtimeCopy(const CopyParams& params) {
    params.check();
    auto         stream_raw  = at::cuda::getCurrentCUDAStream().stream();
    auto         comm_stream = getOverlapStream().stream();
    bool         use_overlap = getEnableCommOverlap();
    cudaStream_t stream      = (params.overlapped && use_overlap) ? comm_stream : stream_raw;

    const auto& src = params.src;
    const auto& dst = params.dst;

    if (src.data_ptr() == dst.data_ptr()) {
        return;
    }

    cudaMemcpyKind copyType;
    if (src.is_cuda() && !dst.is_cuda()) {
        copyType = cudaMemcpyDeviceToHost;
    } else if (!src.is_cuda() && dst.is_cuda()) {
        copyType = cudaMemcpyHostToDevice;
    } else if (src.is_cuda() && dst.is_cuda()) {
        copyType = cudaMemcpyDeviceToDevice;
    } else {
        copyType = cudaMemcpyHostToHost;
    }

    if (copyType == cudaMemcpyHostToHost) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), src.nbytes());
    } else {
        RTP_LLM_DEVICE_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), src.nbytes(), copyType, stream));
    }

    if (copyType == cudaMemcpyDeviceToHost) {
        RTP_LLM_DEVICE_CHECK(cudaStreamSynchronize(stream));
    }

    RTP_LLM_DEVICE_CHECK_DEBUG(stream);
}

static void multiMergeCopyImpl(const MultiMergeCopyParams& params) {
    auto                cur_stream = at::cuda::getCurrentCUDAStream().stream();
    std::vector<void*>  multi_src_ptrs(params.src_ptrs.size());
    std::vector<size_t> multi_src_copy_sizes(params.src_ptrs.size());
    for (size_t i = 0; i < params.src_ptrs.size(); i++) {
        multi_src_ptrs[i]       = params.src_ptrs[i];
        multi_src_copy_sizes[i] = params.copy_size[i];
    }
    InvokeMultiMergeCopyKernel(params.dst_ptr, multi_src_ptrs, multi_src_copy_sizes, params.dst_offsets, cur_stream);
}

static void batchCopyFallback(const BatchCopyParams& params) {
    for (uint32_t copy_type_enum = 0; copy_type_enum < BatchCopyParams::TYPE_SIZE; ++copy_type_enum) {
        auto   copy_type       = BatchCopyParams::CopyType(copy_type_enum);
        auto&  buffers         = params.copy_buffers[copy_type];
        size_t copy_batch_size = buffers.sizes.size();
        if (copy_batch_size == 0) {
            continue;
        }

        for (size_t i = 0; i < copy_batch_size; ++i) {
            size_t        bytes      = buffers.sizes[i];
            torch::Device dst_device = torch::kCPU, src_device = torch::kCPU;
            switch (copy_type) {
                case BatchCopyParams::D2D:
                    dst_device = torch::kCUDA;
                    src_device = torch::kCUDA;
                    break;
                case BatchCopyParams::D2H:
                    dst_device = torch::kCPU;
                    src_device = torch::kCUDA;
                    break;
                case BatchCopyParams::H2D:
                    dst_device = torch::kCUDA;
                    src_device = torch::kCPU;
                    break;
                case BatchCopyParams::H2H:
                    break;
                default:
                    RTP_LLM_FAIL("Unexpected CopyType %d", copy_type);
                    break;
            }
            auto dst_tensor =
                torch::from_blob(buffers.dst_ptr[i], {(int64_t)bytes}, torch::dtype(torch::kUInt8).device(dst_device));
            auto src_tensor = torch::from_blob(const_cast<void*>(buffers.src_ptr[i]),
                                               {(int64_t)bytes},
                                               torch::dtype(torch::kUInt8).device(src_device));
            runtimeCopy({dst_tensor, src_tensor, params.overlapped});
        }
    }
}

void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask) {
    size_t batch_size = logits.size(0);
    size_t vocab_size = logits.size(1);
    auto   dtype      = logits.scalar_type();
    auto   stream     = at::cuda::getCurrentCUDAStream().stream();
    if (dtype == torch::kFloat32) {
        invokeMaskLogits<float>(
            (float*)(logits.data_ptr()), (const uint8_t*)mask.data_ptr(), batch_size, vocab_size, stream);
    } else if (dtype == torch::kFloat16) {
        invokeMaskLogits<half>(
            (half*)(logits.data_ptr()), (const uint8_t*)mask.data_ptr(), batch_size, vocab_size, stream);
    } else if (dtype == torch::kBFloat16) {
        invokeMaskLogits<__nv_bfloat16>(
            (__nv_bfloat16*)(logits.data_ptr()), (const uint8_t*)mask.data_ptr(), batch_size, vocab_size, stream);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  const torch::Tensor& row_indices,
                                  size_t               vocab_size) {
    if (logits.device().is_cpu()) {
        applyPackedMaskLogitsCpuFallback(logits, packed_allow_mask, row_indices, vocab_size);
        return;
    }
    launchPackedMaskLogits(logits,
                           packed_allow_mask,
                           row_indices,
                           vocab_size,
                           at::cuda::getCurrentCUDAStream(logits.device().index()).stream());
}

void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  size_t               vocab_size) {
    runtimeApplyPackedMaskLogits(logits, packed_allow_mask, torch::Tensor{}, vocab_size);
}

#elif USING_ROCM

// ============================================================
// Copy ops (ROCm)
// ============================================================

void runtimeCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
    if (src.data_ptr() == dst.data_ptr()) {
        return;
    }
    // ROCm: dst.copy_(src) dispatches through PyTorch which uses the current HIP stream.
    // params.overlapped is intentionally ignored — ROCm lacks the dedicated overlap stream
    // used by the CUDA path. The default stream provides correct ordering.
    dst.copy_(src, /*non_blocking=*/src.is_hip() && dst.is_hip());
}

static void multiMergeCopyImpl(const MultiMergeCopyParams& params) {
    // MTP passes HIP device pointers for both sources and destination; host
    // memcpy is invalid here, so the ROCm path must launch the device kernel.
    auto                cur_stream = at::hip::getCurrentHIPStream().stream();
    std::vector<void*>  multi_src_ptrs(params.src_ptrs.size());
    std::vector<size_t> multi_src_copy_sizes(params.src_ptrs.size());
    for (size_t i = 0; i < params.src_ptrs.size(); i++) {
        multi_src_ptrs[i]       = params.src_ptrs[i];
        multi_src_copy_sizes[i] = params.copy_size[i];
    }
    InvokeMultiMergeCopyKernel(params.dst_ptr, multi_src_ptrs, multi_src_copy_sizes, params.dst_offsets, cur_stream);
}

static void batchCopyFallback(const BatchCopyParams& params) {
    for (uint32_t copy_type_enum = 0; copy_type_enum < BatchCopyParams::TYPE_SIZE; ++copy_type_enum) {
        auto   copy_type       = BatchCopyParams::CopyType(copy_type_enum);
        auto&  buffers         = params.copy_buffers[copy_type];
        size_t copy_batch_size = buffers.sizes.size();
        if (copy_batch_size == 0)
            continue;

        for (size_t i = 0; i < copy_batch_size; ++i) {
            size_t        bytes      = buffers.sizes[i];
            torch::Device dst_device = torch::kCPU, src_device = torch::kCPU;
            switch (copy_type) {
                case BatchCopyParams::D2D:
                    dst_device = torch::kCUDA;
                    src_device = torch::kCUDA;
                    break;
                case BatchCopyParams::D2H:
                    dst_device = torch::kCPU;
                    src_device = torch::kCUDA;
                    break;
                case BatchCopyParams::H2D:
                    dst_device = torch::kCUDA;
                    src_device = torch::kCPU;
                    break;
                case BatchCopyParams::H2H:
                    break;
                default:
                    RTP_LLM_FAIL("Unexpected CopyType %d", copy_type);
                    break;
            }
            auto dst_tensor =
                torch::from_blob(buffers.dst_ptr[i], {(int64_t)bytes}, torch::dtype(torch::kUInt8).device(dst_device));
            auto src_tensor = torch::from_blob(const_cast<void*>(buffers.src_ptr[i]),
                                               {(int64_t)bytes},
                                               torch::dtype(torch::kUInt8).device(src_device));
            runtimeCopy({dst_tensor, src_tensor, params.overlapped});
        }
    }
}

void runtimeMaskLogits(torch::Tensor& /*logits*/, const torch::Tensor& /*mask*/) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  const torch::Tensor& row_indices,
                                  size_t               vocab_size) {
    applyPackedMaskLogitsCpuFallback(logits, packed_allow_mask, row_indices, vocab_size);
}

void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  size_t               vocab_size) {
    runtimeApplyPackedMaskLogits(logits, packed_allow_mask, torch::Tensor{}, vocab_size);
}

#else

// ============================================================
// Copy ops (CPU / unsupported accelerator builds)
// ============================================================

void runtimeCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
    RTP_LLM_CHECK_WITH_INFO(!src.is_cuda() && !dst.is_cuda(), "runtimeCopy requires host tensors in a non-GPU build");
    if (src.data_ptr() != dst.data_ptr()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), src.nbytes());
    }
}

static void multiMergeCopyImpl(const MultiMergeCopyParams& params) {
    for (size_t i = 0; i < params.src_ptrs.size(); ++i) {
        auto* dst = static_cast<char*>(params.dst_ptr) + params.dst_offsets[i];
        std::memcpy(dst, params.src_ptrs[i], params.copy_size[i]);
    }
}

static void batchCopyFallback(const BatchCopyParams& params) {
    for (uint32_t copy_type_enum = 0; copy_type_enum < BatchCopyParams::TYPE_SIZE; ++copy_type_enum) {
        const auto  copy_type = BatchCopyParams::CopyType(copy_type_enum);
        const auto& buffers   = params.copy_buffers[copy_type];
        if (buffers.sizes.empty()) {
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(copy_type == BatchCopyParams::H2H,
                                "runtimeBatchCopy only supports H2H copies in a non-GPU build");
        for (size_t i = 0; i < buffers.sizes.size(); ++i) {
            std::memcpy(buffers.dst_ptr[i], buffers.src_ptr[i], buffers.sizes[i]);
        }
    }
}

void runtimeMaskLogits(torch::Tensor& /*logits*/, const torch::Tensor& /*mask*/) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  const torch::Tensor& row_indices,
                                  size_t               vocab_size) {
    applyPackedMaskLogitsCpuFallback(logits, packed_allow_mask, row_indices, vocab_size);
}

void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  size_t               vocab_size) {
    runtimeApplyPackedMaskLogits(logits, packed_allow_mask, torch::Tensor{}, vocab_size);
}

#endif

#if USING_CUDA || USING_ROCM
void runtimeBatchCopy(const BatchCopyParams& params) {
    constexpr size_t cuda_sector_size = 128;

    constexpr auto align_to = [](size_t size, size_t alignment) {
        return ((size + alignment - 1) / alignment) * alignment;
    };

#if USING_CUDA
    auto stream_raw = at::cuda::getCurrentCUDAStream().stream();
#else
    auto stream_raw = at::hip::getCurrentHIPStream(at::hip::current_device()).stream();
#endif
    auto         comm_stream = getOverlapStream().stream();
    bool         use_overlap = getEnableCommOverlap();
    cudaStream_t stream      = (params.overlapped && use_overlap) ? comm_stream : stream_raw;

    BatchCopyParams fallback_copies;
    bool            need_fallback = false;

    for (uint32_t copy_type_enum = 0; copy_type_enum < BatchCopyParams::TYPE_SIZE; ++copy_type_enum) {
        auto   copy_type       = BatchCopyParams::CopyType(copy_type_enum);
        auto&  buffers         = params.copy_buffers[copy_type];
        size_t copy_batch_size = buffers.sizes.size();
        if (copy_batch_size == 0) {
            continue;
        }

        switch (copy_type) {
            case BatchCopyParams::D2D: {
                const size_t org_src_ptrs_bytes = sizeof(void*) * copy_batch_size;
                const size_t org_dst_ptrs_bytes = sizeof(void*) * copy_batch_size;
                const size_t org_sizes_bytes    = sizeof(uint64_t) * copy_batch_size;
                const size_t src_ptrs_bytes     = align_to(org_src_ptrs_bytes, cuda_sector_size);
                const size_t dst_ptrs_bytes     = align_to(org_dst_ptrs_bytes, cuda_sector_size);
                const size_t sizes_bytes        = org_sizes_bytes;
                const size_t workspace_bytes    = src_ptrs_bytes + dst_ptrs_bytes + sizes_bytes;

                auto workspace = torch::empty({(int64_t)workspace_bytes},
                                              torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

                auto src_ptrs = reinterpret_cast<void**>(workspace.data_ptr<uint8_t>());
                auto dst_ptrs = reinterpret_cast<void**>(workspace.data_ptr<uint8_t>() + src_ptrs_bytes);
                auto sizes =
                    reinterpret_cast<uint64_t*>(workspace.data_ptr<uint8_t>() + src_ptrs_bytes + dst_ptrs_bytes);

                RTP_LLM_DEVICE_CHECK(cudaMemcpyAsync(
                    src_ptrs, buffers.src_ptr.data(), org_src_ptrs_bytes, cudaMemcpyHostToDevice, stream));
                RTP_LLM_DEVICE_CHECK(cudaMemcpyAsync(
                    dst_ptrs, buffers.dst_ptr.data(), org_dst_ptrs_bytes, cudaMemcpyHostToDevice, stream));
                RTP_LLM_DEVICE_CHECK(
                    cudaMemcpyAsync(sizes, buffers.sizes.data(), org_sizes_bytes, cudaMemcpyHostToDevice, stream));

                cudaEvent_t copy_params_done;
                RTP_LLM_DEVICE_CHECK(cudaEventCreate(&copy_params_done));
                RTP_LLM_DEVICE_CHECK(cudaEventRecord(copy_params_done, stream));

                auto config = kernels::getBatchCopyConfig(buffers.sizes.data(), copy_batch_size);
                kernels::invokeBatchCopy(dst_ptrs, src_ptrs, sizes, copy_batch_size, config, stream);

                RTP_LLM_DEVICE_CHECK(cudaEventSynchronize(copy_params_done));
                RTP_LLM_DEVICE_CHECK(cudaEventDestroy(copy_params_done));
                // The batch-copy kernel reads pointer/size tables from the temporary workspace above.
                // Keep the workspace alive until the copy is complete before returning to the caller.
                RTP_LLM_DEVICE_CHECK(cudaStreamSynchronize(stream));
                RTP_LLM_DEVICE_CHECK_DEBUG(stream);
            } break;
            case BatchCopyParams::H2H:
            case BatchCopyParams::H2D:
            case BatchCopyParams::D2H: {
                need_fallback                           = true;
                fallback_copies.overlapped              = params.overlapped;
                fallback_copies.copy_buffers[copy_type] = buffers;
            } break;
            default:
                RTP_LLM_FAIL("Unexpected CopyType %d", copy_type);
                break;
        }
    }

    if (need_fallback) {
        batchCopyFallback(fallback_copies);
    }
}
#else
void runtimeBatchCopy(const BatchCopyParams& params) {
    batchCopyFallback(params);
}
#endif

// ============================================================
// Non-blocking and fused copies (cross-platform)
// ============================================================

void runtimeNoBlockCopy(const CopyParams& params) {
    params.check();
    const auto& src = params.src;
    const auto& dst = params.dst;
#if USING_CUDA
    const auto  copy_device = getCopyDevice(dst, src);
    DeviceGuard device_guard(copy_device);
    auto        stream = getNoBlockCopyStream(copy_device).stream();
    RTP_LLM_DEVICE_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), src.nbytes(), cudaMemcpyDefault, stream));
    RTP_LLM_DEVICE_CHECK(cudaStreamSynchronize(stream));
    RTP_LLM_DEVICE_CHECK_DEBUG(stream);
#else
    dst.copy_(src);
#endif
}

void runtimeNoBlockCopy(const MultiCopyParams& params) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

#if USING_CUDA
    const bool has_cuda_tensor =
        !params.multi_dst.empty() && (params.multi_dst[0].is_cuda() || params.multi_src[0].is_cuda());
    const int   copy_device = params.multi_dst.empty() ? static_cast<int>(at::cuda::current_device()) :
                                                         getCopyDevice(params.multi_dst[0], params.multi_src[0]);
    DeviceGuard device_guard(copy_device);
    auto        stream = getNoBlockCopyStream(copy_device).stream();

    if (params.split_kv_layer_num > 0 && has_cuda_tensor
        && splitKvMultiCopy(params.multi_src,
                            params.multi_dst,
                            params.split_kv_layer_num,
                            static_cast<int64_t>(params.split_kv_cache_stride_bytes),
                            static_cast<int64_t>(params.split_kv_scale_stride_bytes),
                            stream)) {
        RTP_LLM_DEVICE_CHECK(cudaStreamSynchronize(stream));
        RTP_LLM_DEVICE_CHECK_DEBUG(stream);
        return;
    }

    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        RTP_LLM_DEVICE_CHECK(cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                                             params.multi_src[i].data_ptr(),
                                             params.multi_src[i].nbytes(),
                                             cudaMemcpyDefault,
                                             stream));
    }
    RTP_LLM_DEVICE_CHECK(cudaStreamSynchronize(stream));
    RTP_LLM_DEVICE_CHECK_DEBUG(stream);
#else
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        params.multi_dst[i].copy_(params.multi_src[i]);
    }
#endif
}

void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch& scratch) {
#if USING_CUDA
    if (scratch.device_index >= 0) {
        (void)cudaSetDevice(scratch.device_index);
    }
    if (scratch.host_staging != nullptr) {
        (void)cudaFreeHost(scratch.host_staging);
    }
    releaseDevicePointer(scratch.device_staging);
    releaseMetadataScratch(scratch);
#endif
    scratch.host_staging    = nullptr;
    scratch.host_capacity   = 0;
    scratch.device_staging  = nullptr;
    scratch.device_capacity = 0;
    scratch.device_ptrs     = nullptr;
    scratch.device_offsets  = nullptr;
    scratch.device_sizes    = nullptr;
    scratch.meta_capacity   = 0;
    scratch.device_index    = -1;
}

bool runtimeBatchedMemoryCopy(const BatchedMemoryCopyParams& params) {
    if (params.tiles.empty()) {
        return true;
    }
#if USING_CUDA
    if (params.device_index < 0) {
        RTP_LLM_LOG_WARNING("runtimeBatchedMemoryCopy failed: invalid device_index=%d", params.device_index);
        return false;
    }

#if CUDART_VERSION >= 12080
    DeviceGuard device_guard(params.device_index);
    auto        stream = getNoBlockCopyStream(params.device_index).stream();

    std::vector<void*>       dsts;
    std::vector<const void*> srcs;
    std::vector<size_t>      sizes;
    dsts.reserve(params.tiles.size());
    srcs.reserve(params.tiles.size());
    sizes.reserve(params.tiles.size());
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
    auto   err = cudaMemcpyBatchAsync(
        dsts.data(), mutable_srcs.data(), sizes.data(), dsts.size(), &attr, &attr_idx, 1, &fail_idx, stream);
#endif
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING(
            "runtimeBatchedMemoryCopy failed: tiles=%zu, error=%s", dsts.size(), cudaGetErrorString(err));
        return false;
    }
    RTP_LLM_DEVICE_CHECK_DEBUG(stream);
    return true;
#else
    RTP_LLM_LOG_DEBUG("runtimeBatchedMemoryCopy unavailable: CUDART_VERSION=%d", CUDART_VERSION);
    return false;
#endif
#else
    return false;
#endif
}

bool runtimeStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch* scratch) {
    if (params.tiles.empty()) {
        return true;
    }
#if USING_CUDA
    if (params.device_index < 0 || params.host_bytes == 0 || !checkHostSegments(params)) {
        RTP_LLM_LOG_WARNING("runtimeStagedMemoryCopy failed: device=%d host_base=%p host_bytes=%zu host_segments=%zu",
                            params.device_index,
                            params.host_base,
                            params.host_bytes,
                            params.host_segments.size());
        return false;
    }
    if (checkHostCoverage(params) == HostCoverage::Invalid) {
        RTP_LLM_LOG_WARNING("runtimeStagedMemoryCopy failed: invalid/overlapping host coverage, tiles=%zu bytes=%zu",
                            params.tiles.size(),
                            params.host_bytes);
        return false;
    }

    DeviceGuard device_guard(params.device_index);
    auto        stream = getNoBlockCopyStream(params.device_index).stream();

    std::vector<void*>  host_ptrs;
    std::vector<size_t> host_offsets;
    std::vector<size_t> host_sizes;
    host_ptrs.reserve(params.tiles.size());
    host_offsets.reserve(params.tiles.size());
    host_sizes.reserve(params.tiles.size());
    for (const auto& tile : params.tiles) {
        if (tile.gpu == nullptr || tile.bytes == 0) {
            continue;
        }
        if (tile.host_offset > params.host_bytes || tile.bytes > params.host_bytes - tile.host_offset) {
            RTP_LLM_LOG_WARNING("runtimeStagedMemoryCopy failed: tile out of host span, off=%zu bytes=%zu host=%zu",
                                tile.host_offset,
                                tile.bytes,
                                params.host_bytes);
            return false;
        }
        host_ptrs.push_back(tile.gpu);
        host_offsets.push_back(tile.host_offset);
        host_sizes.push_back(tile.bytes);
    }
    if (host_ptrs.empty()) {
        return true;
    }

    StagedMemoryCopyScratch local_scratch;
    auto*                   work_scratch = scratch != nullptr ? scratch : &local_scratch;
    const auto cleanup_local_scratch = [&]() {
        if (scratch == nullptr) {
            releaseStagedMemoryCopyScratch(local_scratch);
        }
    };

    const size_t tile_num = host_ptrs.size();
    if (!ensureStagedMemoryCopyScratch(*work_scratch, params.device_index, params.host_bytes, tile_num)) {
        cleanup_local_scratch();
        return false;
    }

    auto err = cudaMemcpyAsync(
        work_scratch->device_ptrs, host_ptrs.data(), tile_num * sizeof(void*), cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(work_scratch->device_offsets,
                              host_offsets.data(),
                              tile_num * sizeof(size_t),
                              cudaMemcpyHostToDevice,
                              stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(work_scratch->device_sizes,
                              host_sizes.data(),
                              tile_num * sizeof(size_t),
                              cudaMemcpyHostToDevice,
                              stream);
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
        RTP_LLM_LOG_WARNING("runtimeStagedMemoryCopy failed: tiles=%zu bytes=%zu direction=%s error=%s",
                            tile_num,
                            params.host_bytes,
                            params.direction == StagedMemoryCopyDirection::H2D ? "H2D" : "D2H",
                            cudaGetErrorString(err));
        cleanup_local_scratch();
        return false;
    }
    cleanup_local_scratch();
    RTP_LLM_DEVICE_CHECK_DEBUG(stream);
    return true;
#else
    return false;
#endif
}

void runtimeWarmupNoBlockCopy() {
#if USING_CUDA
    if (!warmupSplitKvCopyKernels(at::cuda::getCurrentCUDAStream().stream())) {
        RTP_LLM_LOG_WARNING("warmupSplitKvCopyKernels failed; split-KV copy may JIT on first use");
    }
#endif
}

void runtimeMultiMergeCopy(const MultiMergeCopyParams& params) {
    multiMergeCopyImpl(params);
}

void fusedCopy(const FusedD2DCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();
    invokeFusedCopy(params, stream);
#else
    throw std::runtime_error("No supported GPU backend found for fusedCopy");
#endif
}

void fusedStridedCopy(const FusedStridedCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedStridedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();
    invokeFusedStridedCopy(params, stream);
#else
    throw std::runtime_error("No supported GPU backend found for fusedStridedCopy");
#endif
}

}  // namespace rtp_llm
