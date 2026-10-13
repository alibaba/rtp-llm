#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"

#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace rtp_llm {

namespace {

__global__ void buildAttentionInputMetadataKernel(const int32_t* __restrict__ input_lengths,
                                                  const int32_t* __restrict__ prefix_lengths,
                                                  int32_t* __restrict__ cu_seqlens,
                                                  int32_t* __restrict__ cu_kv_seqlens,
                                                  int32_t* __restrict__ padding_offset,
                                                  int32_t batch_size,
                                                  int32_t total_tokens) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int32_t max_input_len = 0;
    int32_t q_acc         = 0;
    int32_t kv_acc        = 0;
    cu_seqlens[0]         = 0;
    cu_kv_seqlens[0]      = 0;

    for (int32_t b = 0; b < batch_size; ++b) {
        const int32_t input_len  = input_lengths[b];
        const int32_t prefix_len = prefix_lengths ? prefix_lengths[b] : 0;
        max_input_len            = max_input_len > input_len ? max_input_len : input_len;
        q_acc += input_len;
        kv_acc += input_len + prefix_len;
        cu_seqlens[b + 1]    = q_acc;
        cu_kv_seqlens[b + 1] = kv_acc;
    }

    if (!padding_offset || total_tokens <= 0) {
        return;
    }

    int32_t out_idx    = 0;
    int32_t cum_offset = 0;
    for (int32_t b = 0; b < batch_size; ++b) {
        const int32_t input_len = input_lengths[b];
        for (int32_t j = 0; j < input_len && out_idx < total_tokens; ++j) {
            padding_offset[out_idx++] = cum_offset;
        }
        cum_offset += max_input_len - input_len;
    }
}

}  // namespace

void invokeBuildAttentionInputMetadata(const at::Tensor& input_lengths,
                                       const at::Tensor& prefix_lengths,
                                       at::Tensor&       cu_seqlens,
                                       at::Tensor&       cu_kv_seqlens,
                                       at::Tensor&       padding_offset,
                                       cudaStream_t      stream) {
    TORCH_CHECK(input_lengths.defined(), "input_lengths must be defined");
    TORCH_CHECK(input_lengths.is_cuda(), "input_lengths must be a CUDA tensor");
    TORCH_CHECK(input_lengths.scalar_type() == at::kInt, "input_lengths must be int32");
    TORCH_CHECK(input_lengths.is_contiguous(), "input_lengths must be contiguous");
    TORCH_CHECK(!prefix_lengths.defined() || prefix_lengths.numel() == 0 || prefix_lengths.is_cuda(),
                "prefix_lengths must be CUDA or empty");
    TORCH_CHECK(!prefix_lengths.defined() || prefix_lengths.numel() == 0 || prefix_lengths.scalar_type() == at::kInt,
                "prefix_lengths must be int32");
    TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be CUDA int32");
    TORCH_CHECK(cu_kv_seqlens.is_cuda() && cu_kv_seqlens.scalar_type() == at::kInt, "cu_kv_seqlens must be CUDA int32");
    TORCH_CHECK(!padding_offset.defined() || padding_offset.is_cuda(), "padding_offset must be CUDA");

    const auto batch_size   = static_cast<int32_t>(input_lengths.size(0));
    const auto total_tokens = padding_offset.defined() ? static_cast<int32_t>(padding_offset.numel()) : 0;
    if (batch_size == 0) {
        if (cu_seqlens.numel() > 0) {
            cu_seqlens.zero_();
        }
        if (cu_kv_seqlens.numel() > 0) {
            cu_kv_seqlens.zero_();
        }
        if (padding_offset.defined() && padding_offset.numel() > 0) {
            padding_offset.zero_();
        }
        return;
    }

    const int32_t* prefix_ptr = nullptr;
    if (prefix_lengths.defined() && prefix_lengths.numel() > 0) {
        TORCH_CHECK(prefix_lengths.is_contiguous(), "prefix_lengths must be contiguous");
        TORCH_CHECK(prefix_lengths.size(0) >= batch_size, "prefix_lengths size must cover input_lengths");
        prefix_ptr = prefix_lengths.data_ptr<int32_t>();
    }

    buildAttentionInputMetadataKernel<<<1, 1, 0, stream>>>(
        input_lengths.data_ptr<int32_t>(),
        prefix_ptr,
        cu_seqlens.data_ptr<int32_t>(),
        cu_kv_seqlens.data_ptr<int32_t>(),
        padding_offset.defined() && padding_offset.numel() > 0 ? padding_offset.data_ptr<int32_t>() : nullptr,
        batch_size,
        total_tokens);
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "build attention input metadata kernel failed: ", cudaGetErrorString(result));
}

}  // namespace rtp_llm
