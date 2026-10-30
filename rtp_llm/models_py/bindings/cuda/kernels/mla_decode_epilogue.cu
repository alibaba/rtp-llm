// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from csrc/libtorch_stable/
// fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu in the vLLM source snapshot
// 458ba2edf85bc9b7ebc0d5141f34ea658520faf5 (source SHA256:
// f117b0bcb526e0060e12d1ecb4efbb72255cd3a5e9a4ea7f7ff749a4a892bcb4).
// Changes: retain only BF16 query concat + plain cache insertion without RoPE;
// use RTP ATen tensors, device/stream handling, metadata checks and binding.
// The original vector copies, row splitting and PDL ordering are preserved.
// See the repository LICENSE for the Apache-2.0 license.

#include "rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace rtp_llm {
namespace {

constexpr int kLatentDim = 512;
constexpr int kPositionDim = 64;
constexpr int kEntryDim = kLatentDim + kPositionDim;
constexpr int kVectorElements = 8;  // One uint4 holds eight BF16 elements.
constexpr int kSplitMaxTokens = 64;
constexpr int kBlockThreads = 256;
constexpr int kWarpsPerBlock = kBlockThreads / 32;

template<int RowSplit>
__device__ __forceinline__ void copy_latent_row(at::BFloat16* __restrict__ dst,
                                               const at::BFloat16* __restrict__ latent,
                                               const at::BFloat16* __restrict__ position,
                                               int lane) {
    for (int e = lane * kVectorElements; e < kEntryDim; e += RowSplit * 32 * kVectorElements) {
        const auto* src = e < kLatentDim ? latent + e : position + (e - kLatentDim);
        *reinterpret_cast<uint4*>(dst + e) = *reinterpret_cast<const uint4*>(src);
    }
}

template<int RowSplit>
__global__ void mla_decode_q_concat_cache_insert_kernel(
    const at::BFloat16* __restrict__ q_latent, int64_t q_token_stride, int64_t q_head_stride,
    const at::BFloat16* __restrict__ q_position, int64_t qp_token_stride, int64_t qp_head_stride,
    const at::BFloat16* __restrict__ kv_latent, int64_t kv_token_stride,
    const at::BFloat16* __restrict__ k_position, int64_t kp_token_stride,
    at::BFloat16* __restrict__ q_out, int64_t out_token_stride, int64_t out_head_stride,
    at::BFloat16* __restrict__ cache, int64_t cache_block_stride, int64_t cache_token_stride,
    const int64_t* __restrict__ slot_mapping, int num_tokens, int num_heads, int page_tokens) {
    const int warp = blockIdx.x * kWarpsPerBlock + threadIdx.x / 32;
    const int row = warp / RowSplit;
    const int token = row / (num_heads + 1);
    const int head = row % (num_heads + 1);
    if (token >= num_tokens) {
        return;
    }
    const int lane = (warp % RowSplit) * 32 + threadIdx.x % 32;
    if (head < num_heads) {
        auto* dst = q_out + token * out_token_stride + head * out_head_stride;
        const auto* latent = q_latent + token * q_token_stride + head * q_head_stride;
        const auto* position = q_position + token * qp_token_stride + head * qp_head_stride;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12000
        cudaGridDependencySynchronize();
#endif
        copy_latent_row<RowSplit>(dst, latent, position, lane);
    } else {
        // These inputs precede the query BMM on the current stream and are
        // already ready when PDL overlaps this epilogue with that BMM's tail.
        // Only the query warps need to wait for the BMM's dependent output.
        const int64_t slot = slot_mapping[token];
        if (slot >= 0) {
            auto* dst = cache + (slot / page_tokens) * cache_block_stride
                        + (slot % page_tokens) * cache_token_stride;
            copy_latent_row<RowSplit>(dst, kv_latent + token * kv_token_stride,
                                     k_position + token * kp_token_stride, lane);
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12000
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template<int RowSplit>
void launch_decode_epilogue(const at::Tensor& q_latent, const at::Tensor& q_position,
                           const at::Tensor& kv_latent, const at::Tensor& k_position,
                           at::Tensor& q_out, at::Tensor& cache, const at::Tensor& slot_mapping,
                           cudaStream_t stream, int sm_major) {
    const int64_t warps_per_token = (q_latent.size(1) + 1) * RowSplit;
    TORCH_CHECK(q_latent.size(0) <= (INT_MAX - kWarpsPerBlock) / warps_per_token,
                "MLA epilogue launch is too large");
    const int64_t warps = q_latent.size(0) * warps_per_token;
    const unsigned int blocks = (warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int tokens = static_cast<int>(q_latent.size(0));
    const int heads = static_cast<int>(q_latent.size(1));
    const int page_tokens = static_cast<int>(cache.size(1));

#if CUDART_VERSION >= 12000
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(kBlockThreads);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = sm_major >= 9 ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config, mla_decode_q_concat_cache_insert_kernel<RowSplit>,
        q_latent.data_ptr<at::BFloat16>(), q_latent.stride(0), q_latent.stride(1),
        q_position.data_ptr<at::BFloat16>(), q_position.stride(0), q_position.stride(1),
        kv_latent.data_ptr<at::BFloat16>(), kv_latent.stride(0),
        k_position.data_ptr<at::BFloat16>(), k_position.stride(0),
        q_out.data_ptr<at::BFloat16>(), q_out.stride(0), q_out.stride(1),
        cache.data_ptr<at::BFloat16>(), cache.stride(0), cache.stride(1),
        slot_mapping.data_ptr<int64_t>(), tokens, heads, page_tokens));
#else
    mla_decode_q_concat_cache_insert_kernel<RowSplit><<<blocks, kBlockThreads, 0, stream>>>(
        q_latent.data_ptr<at::BFloat16>(), q_latent.stride(0), q_latent.stride(1),
        q_position.data_ptr<at::BFloat16>(), q_position.stride(0), q_position.stride(1),
        kv_latent.data_ptr<at::BFloat16>(), kv_latent.stride(0),
        k_position.data_ptr<at::BFloat16>(), k_position.stride(0),
        q_out.data_ptr<at::BFloat16>(), q_out.stride(0), q_out.stride(1),
        cache.data_ptr<at::BFloat16>(), cache.stride(0), cache.stride(1),
        slot_mapping.data_ptr<int64_t>(), tokens, heads, page_tokens);
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void mla_decode_q_concat_cache_insert(const at::Tensor& q_latent, const at::Tensor& q_position,
                                     const at::Tensor& kv_latent, const at::Tensor& k_position,
                                     at::Tensor& q_out, at::Tensor& cache, const at::Tensor& slot_mapping) {
    TORCH_CHECK(q_latent.is_cuda() && q_latent.dim() == 3 && q_latent.size(2) == kLatentDim,
                "q_latent must be CUDA [tokens, heads, 512]");
    const int64_t tokens = q_latent.size(0);
    const int64_t heads = q_latent.size(1);
    TORCH_CHECK(tokens <= INT_MAX && heads > 0 && heads < INT_MAX, "Invalid MLA token/head count");
    TORCH_CHECK(q_position.dim() == 3 && q_position.size(0) == tokens && q_position.size(1) == heads
                    && q_position.size(2) == kPositionDim, "q_position must be [tokens, heads, 64]");
    TORCH_CHECK(kv_latent.dim() == 2 && kv_latent.size(0) == tokens && kv_latent.size(1) == kLatentDim,
                "kv_latent must be [tokens, 512]");
    TORCH_CHECK(k_position.dim() == 2 && k_position.size(0) == tokens && k_position.size(1) == kPositionDim,
                "k_position must be [tokens, 64]");
    TORCH_CHECK(q_out.dim() == 3 && q_out.size(0) == tokens && q_out.size(1) == heads
                    && q_out.size(2) == kEntryDim, "q_out must be [tokens, heads, 576]");
    TORCH_CHECK(cache.dim() == 3 && (cache.size(1) == 64 || cache.size(1) == 128)
                    && cache.size(2) == kEntryDim, "cache must be [blocks, page64 or page128, 576]");
    TORCH_CHECK(slot_mapping.device() == q_latent.device() && slot_mapping.scalar_type() == at::kLong
                    && slot_mapping.dim() == 1 && slot_mapping.size(0) == tokens && slot_mapping.is_contiguous(),
                "slot_mapping must be contiguous int64 [tokens] on the query device");
    const at::Tensor* tensors[] = {&q_latent, &q_position, &kv_latent, &k_position, &q_out, &cache};
    for (const at::Tensor* tensor : tensors) {
        TORCH_CHECK(tensor->device() == q_latent.device() && tensor->scalar_type() == at::kBFloat16,
                    "MLA epilogue tensors must be BF16 on the query device");
        TORCH_CHECK(tensor->stride(tensor->dim() - 1) == 1, "MLA epilogue requires unit last-dimension stride");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor->data_ptr()) % sizeof(uint4) == 0,
                    "MLA epilogue requires 16-byte aligned tensor data");
        for (int d = 0; d < tensor->dim() - 1; ++d) {
            TORCH_CHECK(tensor->stride(d) % kVectorElements == 0,
                        "MLA epilogue requires outer strides divisible by eight BF16 elements");
        }
    }
    if (tokens == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(q_latent.device());
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major >= 8, "BF16 MLA epilogue requires SM80 or newer");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q_latent.get_device());
    if (tokens <= kSplitMaxTokens) {
        launch_decode_epilogue<3>(q_latent, q_position, kv_latent, k_position,
                                 q_out, cache, slot_mapping, stream, properties->major);
    } else {
        launch_decode_epilogue<1>(q_latent, q_position, kv_latent, k_position,
                                 q_out, cache, slot_mapping, stream, properties->major);
    }
}

}  // namespace rtp_llm
