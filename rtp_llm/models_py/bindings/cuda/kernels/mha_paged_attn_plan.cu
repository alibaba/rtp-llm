#include "rtp_llm/models_py/bindings/cuda/kernels/mha_paged_attn_plan.h"

#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace rtp_llm {

namespace {

constexpr int kMaxBatchPerCta = 1024;

// One CTA, batch_size threads. Per-batch metadata is staged in shared memory;
// thread 0 does the sequential prefix-sum (batch_size <= 1024 keeps this fast
// and graph-capture-stable); each thread then writes its own token + page slice.
__global__ void mhaPagedAttnPlanKernel(const int32_t* __restrict__ input_lengths,
                                       const int32_t* __restrict__ sequence_lengths,
                                       const int32_t* __restrict__ prefix_lengths,
                                       const int32_t* __restrict__ kv_cache_block_id,
                                       int32_t input_batch_size,
                                       int32_t planned_batch_size,
                                       int32_t max_blocks_per_bs,
                                       int32_t seq_size_per_block,
                                       int32_t* __restrict__ paged_kv_last_page_len,
                                       int32_t* __restrict__ decode_page_indptr,
                                       int32_t* __restrict__ page_indice,
                                       int32_t* __restrict__ batch_indice,
                                       int32_t* __restrict__ positions) {
    __shared__ int32_t s_pages[kMaxBatchPerCta];
    __shared__ int32_t s_input_lens[kMaxBatchPerCta];
    __shared__ int32_t s_token_offset[kMaxBatchPerCta];
    __shared__ int32_t s_page_offset[kMaxBatchPerCta];

    const int tid = threadIdx.x;

    if (tid < planned_batch_size) {
        const bool has_input  = tid < input_batch_size;
        int32_t    input_len  = 0;
        int32_t    prefix_len = 0;
        int32_t    seq_len    = 0;
        if (prefix_lengths) {
            input_len  = has_input ? input_lengths[tid] : 0;
            prefix_len = has_input ? prefix_lengths[tid] : 0;
            seq_len    = input_len + prefix_len;
        } else if (has_input) {
            // Decode mode: one new token per batch; positions[tid] is the
            // 0-indexed slot for that new token (== previous sequence length).
            input_len = 1;
            seq_len   = sequence_lengths[tid] + 1;
        }
        // FlashInfer's SM90 planner rejects kv_len == 0. CUDA graph replay
        // keeps the captured batch size, so inactive tail slots need harmless
        // metadata even though their Q output is discarded.
        const int32_t planner_seq_len = seq_len > 0 ? seq_len : 1;
        s_input_lens[tid]             = input_len;
        s_pages[tid]                  = (planner_seq_len + seq_size_per_block - 1) / seq_size_per_block;
        paged_kv_last_page_len[tid]   = (planner_seq_len - 1) % seq_size_per_block + 1;
    }
    __syncthreads();

    // Sequential exclusive scans over s_input_lens and s_pages.
    if (tid == 0) {
        int32_t t_off         = 0;
        int32_t p_off         = 0;
        decode_page_indptr[0] = 0;
        for (int b = 0; b < planned_batch_size; ++b) {
            s_token_offset[b] = t_off;
            s_page_offset[b]  = p_off;
            t_off += s_input_lens[b];
            p_off += s_pages[b];
            decode_page_indptr[b + 1] = p_off;
        }
    }
    __syncthreads();

    if (tid < planned_batch_size) {
        const bool    has_input  = tid < input_batch_size;
        const int32_t input_len  = s_input_lens[tid];
        const int32_t pages_self = s_pages[tid];
        const int32_t t_start    = s_token_offset[tid];
        const int32_t p_start    = s_page_offset[tid];

        if (prefix_lengths) {
            const int32_t prefix_len = has_input ? prefix_lengths[tid] : 0;
            for (int j = 0; j < input_len; ++j) {
                batch_indice[t_start + j] = tid;
                positions[t_start + j]    = j + prefix_len;
            }
        } else if (has_input) {
            // Decode: single token at the next sequence position.
            batch_indice[t_start] = tid;
            positions[t_start]    = sequence_lengths[tid];
        }

        if (kv_cache_block_id) {
            const bool has_real_kv =
                has_input && (prefix_lengths ? input_lengths[tid] + prefix_lengths[tid] > 0 : true);
            if (!has_real_kv) {
                // Page 0 belongs to the same cache allocation and is safe for
                // ignored graph-padding output. Do not read a nonexistent
                // block-table row for compact replay inputs.
                page_indice[p_start] = 0;
            } else {
                const int32_t* row = kv_cache_block_id + tid * max_blocks_per_bs;
                for (int j = 0; j < pages_self; ++j) {
                    page_indice[p_start + j] = row[j];
                }
            }
        }
    }
}

}  // namespace

void invokeMhaPagedAttnPlan(const at::Tensor& input_lengths,
                            const at::Tensor& sequence_lengths,
                            const at::Tensor& prefix_lengths,
                            const at::Tensor& kv_cache_block_id,
                            int               seq_size_per_block,
                            at::Tensor&       paged_kv_last_page_len,
                            at::Tensor&       decode_page_indptr,
                            at::Tensor&       page_indice,
                            at::Tensor&       batch_indice,
                            at::Tensor&       positions,
                            int               planned_batch_size,
                            cudaStream_t      stream) {
    TORCH_CHECK(input_lengths.defined() && input_lengths.is_cuda() && input_lengths.scalar_type() == at::kInt
                    && input_lengths.is_contiguous(),
                "input_lengths must be a contiguous CUDA int32 tensor");

    const int32_t input_batch_size = static_cast<int32_t>(input_lengths.size(0));
    if (planned_batch_size < 0) {
        planned_batch_size = input_batch_size;
    }
    TORCH_CHECK(planned_batch_size >= input_batch_size,
                "mhaPagedAttnPlan: planned_batch_size ",
                planned_batch_size,
                " is smaller than input batch size ",
                input_batch_size);
    if (planned_batch_size == 0) {
        return;
    }
    TORCH_CHECK(planned_batch_size <= kMaxBatchPerCta,
                "mhaPagedAttnPlan: planned_batch_size ",
                planned_batch_size,
                " exceeds the single-CTA limit ",
                kMaxBatchPerCta);

    const bool has_prefix = prefix_lengths.defined() && prefix_lengths.numel() > 0;
    const bool has_seq    = sequence_lengths.defined() && sequence_lengths.numel() > 0;
    TORCH_CHECK(has_prefix || has_seq, "mhaPagedAttnPlan: need either prefix_lengths or sequence_lengths");

    const int32_t* prefix_ptr = nullptr;
    if (has_prefix) {
        TORCH_CHECK(prefix_lengths.is_cuda() && prefix_lengths.scalar_type() == at::kInt
                        && prefix_lengths.is_contiguous() && prefix_lengths.size(0) >= input_batch_size,
                    "prefix_lengths must be a contiguous CUDA int32 tensor sized >= input batch size");
        prefix_ptr = prefix_lengths.data_ptr<int32_t>();
    }
    const int32_t* seq_ptr = nullptr;
    if (!has_prefix) {
        TORCH_CHECK(
            has_seq && sequence_lengths.is_cuda() && sequence_lengths.scalar_type() == at::kInt
                && sequence_lengths.is_contiguous() && sequence_lengths.size(0) >= input_batch_size,
            "sequence_lengths must be a contiguous CUDA int32 tensor sized >= input batch size when prefix_lengths is empty");
        seq_ptr = sequence_lengths.data_ptr<int32_t>();
    }

    TORCH_CHECK(
        kv_cache_block_id.defined() && kv_cache_block_id.is_cuda() && kv_cache_block_id.scalar_type() == at::kInt
            && kv_cache_block_id.dim() == 2 && kv_cache_block_id.is_contiguous()
            && kv_cache_block_id.size(0) >= input_batch_size,
        "kv_cache_block_id must be a contiguous CUDA int32 [batch, max_blocks] tensor covering the input batch");
    const int32_t max_blocks_per_bs = static_cast<int32_t>(kv_cache_block_id.size(1));
    TORCH_CHECK(max_blocks_per_bs > 0, "kv_cache_block_id must contain at least one page-table column");

    TORCH_CHECK(paged_kv_last_page_len.is_cuda() && paged_kv_last_page_len.scalar_type() == at::kInt
                    && paged_kv_last_page_len.numel() >= planned_batch_size,
                "paged_kv_last_page_len buffer too small");
    TORCH_CHECK(decode_page_indptr.is_cuda() && decode_page_indptr.scalar_type() == at::kInt
                    && decode_page_indptr.numel() >= planned_batch_size + 1,
                "decode_page_indptr buffer too small");
    TORCH_CHECK(page_indice.is_cuda() && page_indice.scalar_type() == at::kInt
                    && page_indice.numel() >= static_cast<int64_t>(planned_batch_size) * max_blocks_per_bs,
                "page_indice buffer too small");
    TORCH_CHECK(batch_indice.is_cuda() && batch_indice.scalar_type() == at::kInt
                    && batch_indice.numel() >= input_batch_size,
                "batch_indice buffer too small");
    TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == at::kInt && positions.numel() >= input_batch_size,
                "positions buffer too small");

    const int threads = ((planned_batch_size + 31) / 32) * 32;  // round up to warp
    mhaPagedAttnPlanKernel<<<1, threads, 0, stream>>>(input_lengths.data_ptr<int32_t>(),
                                                      seq_ptr,
                                                      prefix_ptr,
                                                      kv_cache_block_id.data_ptr<int32_t>(),
                                                      input_batch_size,
                                                      planned_batch_size,
                                                      max_blocks_per_bs,
                                                      seq_size_per_block,
                                                      paged_kv_last_page_len.data_ptr<int32_t>(),
                                                      decode_page_indptr.data_ptr<int32_t>(),
                                                      page_indice.data_ptr<int32_t>(),
                                                      batch_indice.data_ptr<int32_t>(),
                                                      positions.data_ptr<int32_t>());
    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mhaPagedAttnPlanKernel launch failed: ", cudaGetErrorString(err));
}

}  // namespace rtp_llm
