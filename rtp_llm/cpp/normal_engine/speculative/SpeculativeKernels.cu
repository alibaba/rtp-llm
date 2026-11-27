#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeKernels.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace speculative {
namespace {

template<typename T>
__device__ __forceinline__ float toFloat(T v);

template<>
__device__ __forceinline__ float toFloat<float>(float v) {
    return v;
}

template<>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template<typename T>
__device__ __forceinline__ T zeroVal();

template<>
__device__ __forceinline__ float zeroVal<float>() {
    return 0.0f;
}

template<>
__device__ __forceinline__ __nv_bfloat16 zeroVal<__nv_bfloat16>() {
    return __float2bfloat16(0.0f);
}

template<typename T>
__global__ void gatherMarkovEmbeddingKernel(const T* __restrict__ weight,
                                            int64_t w_stride,
                                            const int32_t* __restrict__ indices,
                                            int64_t idx_stride,
                                            T* __restrict__ out,
                                            int64_t vocab_size,
                                            int64_t rank) {
    const int64_t b   = static_cast<int64_t>(blockIdx.y);
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= rank) {
        return;
    }
    const int32_t row = indices[b * idx_stride];
    T             val = zeroVal<T>();
    if (row >= 0 && static_cast<int64_t>(row) < vocab_size) {
        val = weight[static_cast<int64_t>(row) * w_stride + col];
    }
    out[b * rank + col] = val;
}

template<typename BaseT, typename MarkovT>
__global__ void fusedAddMarkovDivTempScalarKernel(const BaseT* __restrict__ base_logits,
                                                  int64_t base_row_stride,
                                                  const MarkovT* __restrict__ markov_bias,
                                                  int64_t markov_row_stride,
                                                  const float* __restrict__ temperature,
                                                  float* __restrict__ out_logits,
                                                  int64_t out_row_stride,
                                                  int64_t step,
                                                  int64_t propose_step,
                                                  int64_t draft_vocab_size) {
    const int64_t b   = static_cast<int64_t>(blockIdx.y);
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= draft_vocab_size) {
        return;
    }
    const BaseT*   base_row   = base_logits + (b * propose_step + step) * base_row_stride;
    const MarkovT* markov_row = markov_bias + b * markov_row_stride;
    float*         out_row    = out_logits + b * out_row_stride;
    const float    temp       = temperature[b];
    out_row[col]              = (toFloat(base_row[col]) + toFloat(markov_row[col])) / temp;
}

__device__ __forceinline__ void loadVec4Float(const float* ptr, float (&vals)[4]) {
    const float4 v = *reinterpret_cast<const float4*>(ptr);
    vals[0]        = v.x;
    vals[1]        = v.y;
    vals[2]        = v.z;
    vals[3]        = v.w;
}

__device__ __forceinline__ void loadVec4Float(const __nv_bfloat16* ptr, float (&vals)[4]) {
    const uint2          raw = *reinterpret_cast<const uint2*>(ptr);
    const __nv_bfloat16* bf  = reinterpret_cast<const __nv_bfloat16*>(&raw);
    vals[0]                  = __bfloat162float(bf[0]);
    vals[1]                  = __bfloat162float(bf[1]);
    vals[2]                  = __bfloat162float(bf[2]);
    vals[3]                  = __bfloat162float(bf[3]);
}

template<typename BaseT, typename MarkovT>
__global__ void fusedAddMarkovDivTempVec4Kernel(const BaseT* __restrict__ base_logits,
                                                int64_t base_row_stride,
                                                const MarkovT* __restrict__ markov_bias,
                                                int64_t markov_row_stride,
                                                const float* __restrict__ temperature,
                                                float* __restrict__ out_logits,
                                                int64_t out_row_stride,
                                                int64_t step,
                                                int64_t propose_step,
                                                int64_t draft_vocab_size) {
    const int64_t b       = static_cast<int64_t>(blockIdx.y);
    const int64_t vec_col = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (vec_col >= draft_vocab_size) {
        return;
    }
    const BaseT*   base_row   = base_logits + (b * propose_step + step) * base_row_stride;
    const MarkovT* markov_row = markov_bias + b * markov_row_stride;
    float*         out_row    = out_logits + b * out_row_stride;
    const float    temp       = temperature[b];

    if (vec_col + 3 < draft_vocab_size) {
        float b_vals[4];
        float m_vals[4];
        loadVec4Float(base_row + vec_col, b_vals);
        loadVec4Float(markov_row + vec_col, m_vals);
        float4 out_vec;
        out_vec.x                                       = (b_vals[0] + m_vals[0]) / temp;
        out_vec.y                                       = (b_vals[1] + m_vals[1]) / temp;
        out_vec.z                                       = (b_vals[2] + m_vals[2]) / temp;
        out_vec.w                                       = (b_vals[3] + m_vals[3]) / temp;
        *reinterpret_cast<float4*>(out_row + vec_col)   = out_vec;
    } else {
        for (int64_t col = vec_col; col < draft_vocab_size; ++col) {
            out_row[col] = (toFloat(base_row[col]) + toFloat(markov_row[col])) / temp;
        }
    }
}

template<typename BaseT, typename MarkovT>
void dispatchDSparkAddMarkovDivTempTypes(const torch::Tensor& base_logits,
                                          const torch::Tensor& markov_bias,
                                          const torch::Tensor& temperature,
                                          torch::Tensor&       out_logits,
                                          int64_t              step,
                                          int64_t              propose_step,
                                          int64_t              draft_vocab_size,
                                          cudaStream_t         stream) {
    const int64_t  batch_size        = temperature.numel();
    const BaseT*   base_ptr          = reinterpret_cast<const BaseT*>(base_logits.data_ptr());
    const MarkovT* markov_ptr        = reinterpret_cast<const MarkovT*>(markov_bias.data_ptr());
    const float*   temp_ptr          = temperature.data_ptr<float>();
    float*         out_ptr           = out_logits.data_ptr<float>();
    const int64_t  base_row_stride   = base_logits.stride(0);
    const int64_t  markov_row_stride = markov_bias.stride(0);
    const int64_t  out_row_stride    = out_logits.stride(0);

    const bool can_vec4 = ((reinterpret_cast<uintptr_t>(base_ptr) & (4 * sizeof(BaseT) - 1)) == 0)
                          && ((reinterpret_cast<uintptr_t>(markov_ptr) & (4 * sizeof(MarkovT) - 1)) == 0)
                          && ((reinterpret_cast<uintptr_t>(out_ptr) & 15) == 0) && (base_row_stride % 4 == 0)
                          && (markov_row_stride % 4 == 0) && (out_row_stride % 4 == 0);

    constexpr int kThreads = 256;
    if (can_vec4) {
        const int64_t vec_cols = (draft_vocab_size + 3) / 4;
        const dim3    grid(static_cast<uint32_t>((vec_cols + kThreads - 1) / kThreads),
                        static_cast<uint32_t>(batch_size));
        fusedAddMarkovDivTempVec4Kernel<BaseT, MarkovT><<<grid, kThreads, 0, stream>>>(base_ptr,
                                                                                       base_row_stride,
                                                                                       markov_ptr,
                                                                                       markov_row_stride,
                                                                                       temp_ptr,
                                                                                       out_ptr,
                                                                                       out_row_stride,
                                                                                       step,
                                                                                       propose_step,
                                                                                       draft_vocab_size);
    } else {
        const dim3 grid(static_cast<uint32_t>((draft_vocab_size + kThreads - 1) / kThreads),
                        static_cast<uint32_t>(batch_size));
        fusedAddMarkovDivTempScalarKernel<BaseT, MarkovT><<<grid, kThreads, 0, stream>>>(base_ptr,
                                                                                         base_row_stride,
                                                                                         markov_ptr,
                                                                                         markov_row_stride,
                                                                                         temp_ptr,
                                                                                         out_ptr,
                                                                                         out_row_stride,
                                                                                         step,
                                                                                         propose_step,
                                                                                         draft_vocab_size);
    }
}

template<typename MapT>
__global__ void mapDraftToTargetInt32Kernel(const int32_t* __restrict__ draft_tokens,
                                            int64_t draft_stride,
                                            const MapT* __restrict__ d2t_map,
                                            int64_t d2t_size,
                                            int32_t* __restrict__ out_tokens,
                                            int64_t out_stride,
                                            int64_t batch_size) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= batch_size) {
        return;
    }
    const int32_t tok = draft_tokens[b * draft_stride];
    const int32_t mapped =
        (tok >= 0 && static_cast<int64_t>(tok) < d2t_size) ? static_cast<int32_t>(d2t_map[tok]) : tok;
    out_tokens[b * out_stride] = mapped;
}

template<typename CapT>
__global__ void postRejectionAndCapKernel(const int32_t* __restrict__ in_tokens,
                                          const int32_t* __restrict__ in_len,
                                          int32_t* __restrict__ out_tokens,
                                          int32_t* __restrict__ out_len,
                                          const bool* __restrict__ force_mask,
                                          const int32_t* __restrict__ draft_tokens,
                                          const int32_t* __restrict__ target_tokens,
                                          int64_t target_row_stride,
                                          int64_t target_col,
                                          const CapT* __restrict__ spec_cap,
                                          int32_t batch_size,
                                          int32_t max_acc_len,
                                          bool    clamp_minus_one) {
    const int32_t b = static_cast<int32_t>(blockIdx.x) * static_cast<int32_t>(blockDim.x)
                      + static_cast<int32_t>(threadIdx.x);
    if (b >= batch_size) {
        return;
    }

    const int32_t propose_step = max_acc_len - 1;
    const bool    forced       = (force_mask != nullptr) && force_mask[b];
    int32_t       len          = forced ? max_acc_len : in_len[b];
    const int64_t tgt_row_base = static_cast<int64_t>(b) * max_acc_len;

    bool    do_cap_replace = false;
    int32_t cap_val        = 0;
    int32_t cap_tok        = 0;
    if (spec_cap != nullptr) {
        const int64_t raw_cap    = static_cast<int64_t>(spec_cap[b]);
        const int64_t capped_len = raw_cap + 1;
        if (capped_len < static_cast<int64_t>(len)) {
            len = static_cast<int32_t>(capped_len);
        }
        if (raw_cap >= 0 && raw_cap < static_cast<int64_t>(propose_step) && static_cast<int64_t>(len) > raw_cap) {
            do_cap_replace = true;
            cap_val        = static_cast<int32_t>(raw_cap);
            cap_tok        = target_tokens[(tgt_row_base + cap_val) * target_row_stride + target_col];
        }
    }

    if (out_len != in_len || forced || spec_cap != nullptr) {
        out_len[b] = len;
    }

    const int64_t row_off   = static_cast<int64_t>(b) * max_acc_len;
    const int64_t draft_off = static_cast<int64_t>(b) * propose_step;
    for (int32_t col = 0; col < max_acc_len; ++col) {
        int32_t tok;
        if (forced) {
            tok = (col < propose_step)
                      ? draft_tokens[draft_off + col]
                      : target_tokens[(tgt_row_base + propose_step) * target_row_stride + target_col];
        } else {
            tok = in_tokens[row_off + col];
        }
        if (clamp_minus_one && tok == -1) {
            tok = 0;
        }
        if (do_cap_replace && col == cap_val) {
            tok = cap_tok;
        }
        out_tokens[row_off + col] = tok;
    }
}

int32_t nextPow2Clamped(int32_t n) {
    int32_t p = 32;
    while (p < n && p < 256) {
        p <<= 1;
    }
    return std::min(p, 256);
}

}  // namespace

void launchGatherMarkovEmbedding(const torch::Tensor& markov_w1,
                                 const torch::Tensor& indices,
                                 torch::Tensor&       out) {
    const int64_t batch_size = indices.numel();
    const int64_t rank       = markov_w1.size(1);
    if (batch_size == 0 || rank == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(markov_w1.is_cuda() && indices.is_cuda() && out.is_cuda(),
                            "gatherMarkovEmbedding requires CUDA tensors");
    RTP_LLM_CHECK_WITH_INFO(markov_w1.dim() == 2 && markov_w1.stride(1) == 1 && out.is_contiguous(),
                            "gatherMarkovEmbedding requires row-contiguous [V, R] tensors");
    RTP_LLM_CHECK_WITH_INFO(indices.dim() == 1 && indices.scalar_type() == torch::kInt32,
                            "gatherMarkovEmbedding requires 1D int32 indices");
    RTP_LLM_CHECK_WITH_INFO(markov_w1.scalar_type() == out.scalar_type(),
                            "gatherMarkovEmbedding requires matching markov_w1 and out dtypes");

    const cudaStream_t stream   = at::cuda::getCurrentCUDAStream().stream();
    constexpr int      kThreads = 128;
    const dim3         grid(static_cast<uint32_t>((rank + kThreads - 1) / kThreads), static_cast<uint32_t>(batch_size));
    const int64_t      vocab_size = markov_w1.size(0);
    const int64_t      w_stride   = markov_w1.stride(0);
    const int64_t      idx_stride = indices.stride(0);
    const auto*        idx_ptr    = indices.data_ptr<int32_t>();

    if (markov_w1.scalar_type() == torch::kBFloat16) {
        gatherMarkovEmbeddingKernel<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(markov_w1.data_ptr()),
            w_stride,
            idx_ptr,
            idx_stride,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            vocab_size,
            rank);
    } else if (markov_w1.scalar_type() == torch::kFloat32) {
        gatherMarkovEmbeddingKernel<float><<<grid, kThreads, 0, stream>>>(
            markov_w1.data_ptr<float>(),
            w_stride,
            idx_ptr,
            idx_stride,
            out.data_ptr<float>(),
            vocab_size,
            rank);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "gatherMarkovEmbedding supports bfloat16 or float32 weights");
    }

    const cudaError_t err = cudaGetLastError();
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess, "gatherMarkovEmbeddingKernel failed: %s", cudaGetErrorString(err));
}

void launchDSparkAddMarkovDivTemp(const torch::Tensor& base_logits,
                                  const torch::Tensor& markov_bias,
                                  const torch::Tensor& temperature,
                                  torch::Tensor&       out_logits,
                                  int64_t              step,
                                  int64_t              propose_step,
                                  int64_t              draft_vocab_size) {
    const int64_t batch_size = temperature.numel();
    if (batch_size == 0 || draft_vocab_size == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(base_logits.is_cuda() && markov_bias.is_cuda() && temperature.is_cuda()
                                && out_logits.is_cuda(),
                            "launchDSparkAddMarkovDivTemp requires CUDA tensors");
    RTP_LLM_CHECK_WITH_INFO(base_logits.dim() == 2 && base_logits.stride(1) == 1 && markov_bias.dim() == 2
                                && markov_bias.stride(1) == 1 && out_logits.dim() == 2 && out_logits.stride(1) == 1,
                            "launchDSparkAddMarkovDivTemp requires row-contiguous 2D tensors");
    RTP_LLM_CHECK_WITH_INFO(temperature.is_contiguous() && temperature.scalar_type() == torch::kFloat32,
                            "launchDSparkAddMarkovDivTemp requires contiguous float32 temperature");
    RTP_LLM_CHECK_WITH_INFO(out_logits.scalar_type() == torch::kFloat32,
                            "launchDSparkAddMarkovDivTemp requires float32 output");

    const cudaStream_t stream      = at::cuda::getCurrentCUDAStream().stream();
    const auto         base_dtype  = base_logits.scalar_type();
    const auto         markov_dtype = markov_bias.scalar_type();

    if (base_dtype == torch::kBFloat16 && markov_dtype == torch::kBFloat16) {
        dispatchDSparkAddMarkovDivTempTypes<__nv_bfloat16, __nv_bfloat16>(
            base_logits, markov_bias, temperature, out_logits, step, propose_step, draft_vocab_size, stream);
    } else if (base_dtype == torch::kFloat32 && markov_dtype == torch::kBFloat16) {
        dispatchDSparkAddMarkovDivTempTypes<float, __nv_bfloat16>(
            base_logits, markov_bias, temperature, out_logits, step, propose_step, draft_vocab_size, stream);
    } else if (base_dtype == torch::kBFloat16 && markov_dtype == torch::kFloat32) {
        dispatchDSparkAddMarkovDivTempTypes<__nv_bfloat16, float>(
            base_logits, markov_bias, temperature, out_logits, step, propose_step, draft_vocab_size, stream);
    } else if (base_dtype == torch::kFloat32 && markov_dtype == torch::kFloat32) {
        dispatchDSparkAddMarkovDivTempTypes<float, float>(
            base_logits, markov_bias, temperature, out_logits, step, propose_step, draft_vocab_size, stream);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false,
                                "launchDSparkAddMarkovDivTemp supports float32/bfloat16 inputs, got %s and %s",
                                c10::toString(base_dtype),
                                c10::toString(markov_dtype));
    }

    const cudaError_t err = cudaGetLastError();
    RTP_LLM_CHECK_WITH_INFO(
        err == cudaSuccess, "fusedAddMarkovDivTempKernel failed: %s", cudaGetErrorString(err));
}

void launchMapDraftToTargetInt32(const torch::Tensor& draft_tokens,
                                   const torch::Tensor& d2t_map,
                                   torch::Tensor&       out_tokens) {
    const int64_t batch_size = draft_tokens.numel();
    if (batch_size == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(draft_tokens.is_cuda() && d2t_map.is_cuda() && out_tokens.is_cuda(),
                            "launchMapDraftToTargetInt32 requires CUDA tensors");
    RTP_LLM_CHECK_WITH_INFO(draft_tokens.dim() == 1 && out_tokens.dim() == 1 && out_tokens.numel() == batch_size
                                && draft_tokens.scalar_type() == torch::kInt32
                                && out_tokens.scalar_type() == torch::kInt32,
                            "launchMapDraftToTargetInt32 requires 1D int32 tokens");
    RTP_LLM_CHECK_WITH_INFO(d2t_map.dim() == 1 && d2t_map.is_contiguous(),
                            "launchMapDraftToTargetInt32 requires contiguous 1D d2t_map");

    const cudaStream_t stream   = at::cuda::getCurrentCUDAStream().stream();
    constexpr int      kThreads = 128;
    const int          blocks   = static_cast<int>((batch_size + kThreads - 1) / kThreads);
    if (d2t_map.scalar_type() == torch::kInt64) {
        mapDraftToTargetInt32Kernel<int64_t><<<blocks, kThreads, 0, stream>>>(draft_tokens.data_ptr<int32_t>(),
                                                                              draft_tokens.stride(0),
                                                                              d2t_map.data_ptr<int64_t>(),
                                                                              d2t_map.numel(),
                                                                              out_tokens.data_ptr<int32_t>(),
                                                                              out_tokens.stride(0),
                                                                              batch_size);
    } else if (d2t_map.scalar_type() == torch::kInt32) {
        mapDraftToTargetInt32Kernel<int32_t><<<blocks, kThreads, 0, stream>>>(draft_tokens.data_ptr<int32_t>(),
                                                                              draft_tokens.stride(0),
                                                                              d2t_map.data_ptr<int32_t>(),
                                                                              d2t_map.numel(),
                                                                              out_tokens.data_ptr<int32_t>(),
                                                                              out_tokens.stride(0),
                                                                              batch_size);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "launchMapDraftToTargetInt32 supports int64 or int32 d2t_map");
    }

    const cudaError_t err = cudaGetLastError();
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess, "mapDraftToTargetInt32Kernel failed: %s", cudaGetErrorString(err));
}

void launchPostRejectionAndCap(const torch::Tensor& in_tokens,
                               const torch::Tensor& in_accept_len,
                               torch::Tensor&       out_tokens,
                               torch::Tensor&       out_accept_len,
                               const torch::Tensor& force_mask,
                               const torch::Tensor& draft_token_ids,
                               const torch::Tensor& target_token_ids,
                               const torch::Tensor& spec_cap,
                               bool                 clamp_minus_one) {
    const int32_t batch_size  = static_cast<int32_t>(in_tokens.size(0));
    const int32_t max_acc_len = static_cast<int32_t>(in_tokens.size(1));
    if (batch_size <= 0 || max_acc_len <= 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(in_tokens.is_cuda() && in_accept_len.is_cuda() && out_tokens.is_cuda()
                                && out_accept_len.is_cuda(),
                            "launchPostRejectionAndCap requires CUDA token and length tensors");
    RTP_LLM_CHECK_WITH_INFO(in_tokens.is_contiguous() && in_accept_len.is_contiguous() && out_tokens.is_contiguous()
                                && out_accept_len.is_contiguous() && in_tokens.scalar_type() == torch::kInt32
                                && in_accept_len.scalar_type() == torch::kInt32
                                && out_tokens.scalar_type() == torch::kInt32
                                && out_accept_len.scalar_type() == torch::kInt32,
                            "launchPostRejectionAndCap requires contiguous int32 tensors");

    const bool has_force = force_mask.defined();
    const bool has_cap   = spec_cap.defined();

    const bool*    force_ptr  = nullptr;
    const int32_t* draft_ptr  = nullptr;
    if (has_force) {
        RTP_LLM_CHECK_WITH_INFO(force_mask.is_cuda() && force_mask.is_contiguous()
                                    && force_mask.scalar_type() == torch::kBool && force_mask.numel() == batch_size,
                                "force_mask must be contiguous CUDA bool [B]");
        RTP_LLM_CHECK_WITH_INFO(draft_token_ids.defined() && draft_token_ids.is_cuda()
                                    && draft_token_ids.is_contiguous()
                                    && draft_token_ids.scalar_type() == torch::kInt32
                                    && draft_token_ids.dim() == 2 && draft_token_ids.size(0) == batch_size
                                    && draft_token_ids.size(1) == max_acc_len - 1,
                                "draft_token_ids must be contiguous CUDA int32 [B, propose_step]");
        force_ptr = force_mask.data_ptr<bool>();
        draft_ptr = draft_token_ids.data_ptr<int32_t>();
    }

    const int32_t* target_ptr        = nullptr;
    int64_t        target_row_stride = 0;
    int64_t        target_col        = 0;
    if (has_force || has_cap) {
        RTP_LLM_CHECK_WITH_INFO(target_token_ids.defined() && target_token_ids.is_cuda()
                                    && target_token_ids.dim() == 2
                                    && target_token_ids.scalar_type() == torch::kInt32
                                    && target_token_ids.size(0) == static_cast<int64_t>(batch_size) * max_acc_len
                                    && target_token_ids.size(1) >= 1,
                                "target_token_ids must be CUDA int32 [B*(propose_step+1), stride]");
        target_ptr        = target_token_ids.data_ptr<int32_t>();
        target_row_stride = target_token_ids.stride(0);
        target_col        = target_token_ids.size(1) - 1;
    }

    const cudaStream_t stream    = at::cuda::getCurrentCUDAStream().stream();
    const int32_t      block_dim = nextPow2Clamped(batch_size);
    const int32_t      grid_dim  = (batch_size + block_dim - 1) / block_dim;

    if (!has_cap) {
        postRejectionAndCapKernel<int32_t><<<grid_dim, block_dim, 0, stream>>>(in_tokens.data_ptr<int32_t>(),
                                                                               in_accept_len.data_ptr<int32_t>(),
                                                                               out_tokens.data_ptr<int32_t>(),
                                                                               out_accept_len.data_ptr<int32_t>(),
                                                                               force_ptr,
                                                                               draft_ptr,
                                                                               target_ptr,
                                                                               target_row_stride,
                                                                               target_col,
                                                                               nullptr,
                                                                               batch_size,
                                                                               max_acc_len,
                                                                               clamp_minus_one);
    } else {
        RTP_LLM_CHECK_WITH_INFO(spec_cap.is_cuda() && spec_cap.is_contiguous() && spec_cap.numel() == batch_size,
                                "spec_cap must be contiguous CUDA [B]");
        if (spec_cap.scalar_type() == torch::kInt32) {
            postRejectionAndCapKernel<int32_t><<<grid_dim, block_dim, 0, stream>>>(in_tokens.data_ptr<int32_t>(),
                                                                                   in_accept_len.data_ptr<int32_t>(),
                                                                                   out_tokens.data_ptr<int32_t>(),
                                                                                   out_accept_len.data_ptr<int32_t>(),
                                                                                   force_ptr,
                                                                                   draft_ptr,
                                                                                   target_ptr,
                                                                                   target_row_stride,
                                                                                   target_col,
                                                                                   spec_cap.data_ptr<int32_t>(),
                                                                                   batch_size,
                                                                                   max_acc_len,
                                                                                   clamp_minus_one);
        } else if (spec_cap.scalar_type() == torch::kInt64) {
            postRejectionAndCapKernel<int64_t><<<grid_dim, block_dim, 0, stream>>>(in_tokens.data_ptr<int32_t>(),
                                                                                   in_accept_len.data_ptr<int32_t>(),
                                                                                   out_tokens.data_ptr<int32_t>(),
                                                                                   out_accept_len.data_ptr<int32_t>(),
                                                                                   force_ptr,
                                                                                   draft_ptr,
                                                                                   target_ptr,
                                                                                   target_row_stride,
                                                                                   target_col,
                                                                                   spec_cap.data_ptr<int64_t>(),
                                                                                   batch_size,
                                                                                   max_acc_len,
                                                                                   clamp_minus_one);
        } else {
            RTP_LLM_CHECK_WITH_INFO(false, "spec_cap must be int32 or int64");
        }
    }

    const cudaError_t err = cudaGetLastError();
    RTP_LLM_CHECK_WITH_INFO(err == cudaSuccess, "postRejectionAndCapKernel failed: %s", cudaGetErrorString(err));
}

}  // namespace speculative
}  // namespace rtp_llm
