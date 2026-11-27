#include "rtp_llm/models_py/bindings/cuda/kernels/attention_input_metadata.h"

#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace rtp_llm {

namespace {

constexpr int32_t kPaddingOffsetBlockSize = 256;
constexpr int32_t kWarpSize               = 32;
constexpr int32_t kNumWarps               = kPaddingOffsetBlockSize / kWarpSize;

__global__ void buildAttentionCuSeqlensKernel(const int32_t* __restrict__ input_lengths,
                                              const int32_t* __restrict__ prefix_lengths,
                                              int32_t* __restrict__ cu_seqlens,
                                              int32_t* __restrict__ cu_kv_seqlens,
                                              int32_t batch_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int32_t q_acc    = 0;
    int32_t kv_acc   = 0;
    cu_seqlens[0]    = 0;
    cu_kv_seqlens[0] = 0;

    for (int32_t b = 0; b < batch_size; ++b) {
        const int32_t input_len  = input_lengths[b];
        const int32_t prefix_len = prefix_lengths ? prefix_lengths[b] : 0;
        q_acc += input_len;
        kv_acc += input_len + prefix_len;
        cu_seqlens[b + 1]    = q_acc;
        cu_kv_seqlens[b + 1] = kv_acc;
    }
}

__global__ void computePaddingOffsetKernel(const int32_t* __restrict__ cu_seqlens,
                                           int32_t* __restrict__ padding_offset,
                                           int32_t batch_size,
                                           int32_t total_tokens) {
    __shared__ int32_t smem_warp_max[kNumWarps];
    __shared__ int32_t smem_max_input_len;

    int32_t local_max = 0;
    for (int32_t b = static_cast<int32_t>(threadIdx.x); b < batch_size; b += kPaddingOffsetBlockSize) {
        const int32_t input_len = cu_seqlens[b + 1] - cu_seqlens[b];
        if (input_len > local_max) {
            local_max = input_len;
        }
    }

    const int32_t lane    = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
    const int32_t warp_id = static_cast<int32_t>(threadIdx.x) / kWarpSize;

    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        const int32_t other = __shfl_down_sync(0xffffffffu, local_max, offset);
        if (other > local_max) {
            local_max = other;
        }
    }
    if (lane == 0) {
        smem_warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        int32_t block_max = (lane < kNumWarps) ? smem_warp_max[lane] : 0;
        for (int offset = kNumWarps >> 1; offset > 0; offset >>= 1) {
            const int32_t other = __shfl_down_sync(0xffffffffu, block_max, offset);
            if (other > block_max) {
                block_max = other;
            }
        }
        if (lane == 0) {
            smem_max_input_len = block_max;
        }
    }
    __syncthreads();

    const int32_t idx          = static_cast<int32_t>(blockIdx.x) * kPaddingOffsetBlockSize + static_cast<int32_t>(threadIdx.x);
    const int32_t valid_tokens = total_tokens < cu_seqlens[batch_size] ? total_tokens : cu_seqlens[batch_size];
    if (idx >= valid_tokens) {
        return;
    }

    // Find the largest batch index b in [0, batch_size - 1] such that cu_seqlens[b] <= idx.
    int32_t lo = 0;
    int32_t hi = batch_size + 1;
    while (lo < hi) {
        const int32_t mid = lo + ((hi - lo) >> 1);
        if (cu_seqlens[mid] <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int32_t b = lo - 1;
    padding_offset[idx] = b * smem_max_input_len - cu_seqlens[b];
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

    int32_t* padding_ptr =
        padding_offset.defined() && padding_offset.numel() > 0 ? padding_offset.data_ptr<int32_t>() : nullptr;

    buildAttentionCuSeqlensKernel<<<1, 1, 0, stream>>>(
        input_lengths.data_ptr<int32_t>(),
        prefix_ptr,
        cu_seqlens.data_ptr<int32_t>(),
        cu_kv_seqlens.data_ptr<int32_t>(),
        batch_size);

    if (padding_ptr != nullptr && total_tokens > 0) {
        const int32_t num_blocks = (total_tokens + kPaddingOffsetBlockSize - 1) / kPaddingOffsetBlockSize;
        computePaddingOffsetKernel<<<num_blocks, kPaddingOffsetBlockSize, 0, stream>>>(
            cu_seqlens.data_ptr<int32_t>(), padding_ptr, batch_size, total_tokens);
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "build attention input metadata kernel failed: ", cudaGetErrorString(result));
}

}  // namespace rtp_llm
