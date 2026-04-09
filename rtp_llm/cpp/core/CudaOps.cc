#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/core/CommonDefines.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include <memory>
#include <unistd.h>

#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/cpp/core/torch_utils/TorchEvent.h"
#include "ATen/ops/cat.h"
#include "rtp_llm/models_py/bindings/common/kernels/batch_copy.h"
#include "rtp_llm/models_py/bindings/common/kernels/copy_utils.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/models_py/bindings/common/kernels/mask_logits.h"
#include <cuda_profiler_api.h>
#elif USING_ROCM
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

using namespace std;

namespace rtp_llm {

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

void runtimeBatchCopy(const BatchCopyParams& params) {
    constexpr size_t cuda_sector_size = 128;

    constexpr auto align_to = [](size_t size, size_t alignment) {
        return ((size + alignment - 1) / alignment) * alignment;
    };

    auto         stream_raw  = at::cuda::getCurrentCUDAStream().stream();
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

#else  // ROCm / non-CUDA

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

void runtimeBatchCopy(const BatchCopyParams& params) {
    batchCopyFallback(params);
}

void runtimeMaskLogits(torch::Tensor& logits, const torch::Tensor& mask) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

#endif  // USING_CUDA

}  // namespace rtp_llm
