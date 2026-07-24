#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/CommonDefines.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include <ATen/Dispatch.h>
#include <limits>
#include <memory>
#include <unistd.h>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "ATen/ops/cat.h"
#include "rtp_llm/models_py/bindings/common/kernels/batch_copy.h"
#include "rtp_llm/models_py/bindings/common/kernels/copy_utils.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"
#include <cuda_profiler_api.h>
#elif USING_ROCM
#include <ATen/hip/HIPContext.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include "rtp_llm/models_py/bindings/common/kernels/batch_copy.h"
#include "rtp_llm/models_py/bindings/common/kernels/copy_utils.h"
#include "rtp_llm/models_py/bindings/rocm/hip_host_utils.h"
#endif

#if USING_CUDA || USING_ROCM
#include "rtp_llm/models_py/bindings/common/kernels/mask_logits.h"
#endif

using namespace std;

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
                        logits_row_data[token] = static_cast<scalar_t>(-std::numeric_limits<float>::infinity());
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

namespace {
at::cuda::CUDAStream& getOverlapStream() {
    static thread_local auto s = at::cuda::getStreamFromPool(/*isHighPriority=*/true);
    return s;
}
}  // anonymous namespace

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
        check_cuda_value(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), src.nbytes(), copyType, stream));
    }

    if (copyType == cudaMemcpyDeviceToHost) {
        check_cuda_value(cudaStreamSynchronize(stream));
    }

    check_cuda_error();
}

void multiMergeCopy(const MultiMergeCopyParams& params) {
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

// ============================================================
// maskLogits (CUDA)
// ============================================================

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
    launchPackedMaskLogits(logits,
                           packed_allow_mask,
                           row_indices,
                           vocab_size,
                           at::cuda::getCurrentCUDAStream(logits.device().index()).stream());
}

void runtimeApplyPackedMaskLogits(const torch::Tensor& logits,
                                  const torch::Tensor& packed_allow_mask,
                                  size_t               vocab_size) {
    launchPackedMaskLogits(logits,
                           packed_allow_mask,
                           torch::Tensor{},
                           vocab_size,
                           at::cuda::getCurrentCUDAStream(logits.device().index()).stream());
}

#else  // ROCm / non-CUDA

namespace {
at::hip::HIPStream& getOverlapStream() {
    static thread_local auto s = at::hip::getStreamFromPool(/*isHighPriority=*/true);
    return s;
}
}  // anonymous namespace

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

void multiMergeCopy(const MultiMergeCopyParams& params) {
    for (size_t i = 0; i < params.src_ptrs.size(); i++) {
        auto dst = static_cast<char*>(params.dst_ptr) + params.dst_offsets[i];
        std::memcpy(dst, params.src_ptrs[i], params.copy_size[i]);
    }
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

void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask) {
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
    applyPackedMaskLogitsCpuFallback(logits, packed_allow_mask, torch::Tensor{}, vocab_size);
}

#endif  // USING_CUDA

#if USING_CUDA || USING_ROCM
void runtimeBatchCopy(const BatchCopyParams& params) {
    constexpr size_t cuda_sector_size = 128;

    constexpr auto align_to = [](size_t size, size_t alignment) {
        return ((size + alignment - 1) / alignment) * alignment;
    };

#if USING_CUDA
    auto stream_raw = at::cuda::getCurrentCUDAStream().stream();
#else   // USING_ROCM
    auto stream_raw = at::hip::getCurrentHIPStream(at::hip::current_device()).stream();
#endif  // USING_CUDA
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

                check_cuda_value(cudaMemcpyAsync(
                    src_ptrs, buffers.src_ptr.data(), org_src_ptrs_bytes, cudaMemcpyHostToDevice, stream));
                check_cuda_value(cudaMemcpyAsync(
                    dst_ptrs, buffers.dst_ptr.data(), org_dst_ptrs_bytes, cudaMemcpyHostToDevice, stream));
                check_cuda_value(
                    cudaMemcpyAsync(sizes, buffers.sizes.data(), org_sizes_bytes, cudaMemcpyHostToDevice, stream));

                cudaEvent_t copy_params_done;
                check_cuda_value(cudaEventCreate(&copy_params_done));
                check_cuda_value(cudaEventRecord(copy_params_done, stream));

                auto config = kernels::getBatchCopyConfig(buffers.sizes.data(), copy_batch_size);
                kernels::invokeBatchCopy(dst_ptrs, src_ptrs, sizes, copy_batch_size, config, stream);

                check_cuda_value(cudaEventSynchronize(copy_params_done));
                check_cuda_value(cudaEventDestroy(copy_params_done));
                // The batch-copy kernel reads pointer/size tables from the temporary workspace above.
                // Keep the workspace alive until the copy is complete before returning to the caller.
                check_cuda_value(cudaStreamSynchronize(stream));

                check_cuda_error();
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

}  // namespace rtp_llm
