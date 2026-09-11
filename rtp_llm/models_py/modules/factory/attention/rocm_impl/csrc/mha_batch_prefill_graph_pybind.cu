#include "mha_common.h"
#include "mha_fwd.h"
#include "py_itfs_common.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace aiter::torch_itfs {

fmha_batch_prefill_args get_ck_fmha_batch_prefill_args(bool             has_lse,
                                                       bool             has_dropout_randval,
                                                       const mask_info& mask,
                                                       int              b,
                                                       int              max_seqlen_q,
                                                       int              h,
                                                       int              h_k,
                                                       int              d,
                                                       int              d_v,
                                                       int              num_total_pages,
                                                       int              page_block_size,
                                                       ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout,
                                                       const at::Tensor                               q,
                                                       const at::Tensor                               k,
                                                       const at::Tensor                               v,
                                                       const at::Tensor                               seqlens_q,
                                                       const at::Tensor                               kv_indptr,
                                                       const at::Tensor                               kv_page_indices,
                                                       std::optional<const at::Tensor>                sink_ptr,
                                                       std::optional<const at::Tensor>&               bias,
                                                       std::optional<const at::Tensor>&               alibi_slopes,
                                                       std::optional<const at::Tensor>&               q_descale,
                                                       std::optional<const at::Tensor>&               k_descale,
                                                       std::optional<const at::Tensor>&               v_descale,
                                                       std::optional<const at::Tensor>&               kv_block_descale,
                                                       at::Tensor                                     out,
                                                       at::Tensor                                     softmax_lse,
                                                       at::Tensor                                     dropout_randval,
                                                       float                                          softmax_scale,
                                                       float                                          logits_soft_cap,
                                                       float                                          p_dropout,
                                                       std::pair<uint64_t*, uint64_t*>                drop_seed_offset,
                                                       std::optional<const at::Tensor>& kv_last_page_lens);

__global__ void sanitize_block_table_kernel(const int32_t* block_table,
                                            const int32_t* seqlen_k,
                                            int32_t*       sanitized_block_table,
                                            int            input_cols,
                                            int            output_cols,
                                            int            tokens_per_block) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= output_cols)
        return;

    int valid_blocks                               = (seqlen_k[row] + tokens_per_block - 1) / tokens_per_block;
    valid_blocks                                   = valid_blocks < 1 ? 1 : valid_blocks;
    valid_blocks                                   = valid_blocks > input_cols ? input_cols : valid_blocks;
    const int source_col                           = col < valid_blocks ? col : valid_blocks - 1;
    sanitized_block_table[row * output_cols + col] = block_table[row * input_cols + source_col];
}

__device__ bool claim_page(int32_t page, int32_t* claims, int capacity) {
    int slot = page % capacity;
    for (int probe = 0; probe < capacity; ++probe) {
        const int32_t previous = atomicCAS(claims + slot, -1, page);
        if (previous == -1)
            return true;
        if (previous == page)
            return false;
        slot = slot + 1 == capacity ? 0 : slot + 1;
    }
    return false;
}

__global__ void transpose_linear_v_pages_kernel(uint16_t*      v,
                                                const int32_t* block_table,
                                                const int32_t* seqlen_k,
                                                int32_t*       page_claims,
                                                int            batch_size,
                                                int            input_cols,
                                                int            num_blocks,
                                                int            num_heads,
                                                int            page_size,
                                                int            head_dim,
                                                int64_t        block_stride,
                                                int64_t        head_stride,
                                                bool           to_vectorized) {
    const int entry = blockIdx.x;
    const int row   = entry / input_cols;
    const int col   = entry % input_cols;
    if (row >= batch_size)
        return;

    int valid_blocks = (seqlen_k[row] + page_size - 1) / page_size;
    valid_blocks     = valid_blocks < 1 ? 1 : valid_blocks;
    valid_blocks     = valid_blocks > input_cols ? input_cols : valid_blocks;
    if (col >= valid_blocks)
        return;

    const int32_t page = block_table[entry];
    if (page < 0 || page >= num_blocks)
        return;

    __shared__ int owner;
    if (threadIdx.x == 0)
        owner = claim_page(page, page_claims, batch_size * input_cols);
    __syncthreads();
    if (!owner)
        return;

    extern __shared__ uint16_t tile[];
    uint16_t*                  page_ptr    = v + static_cast<int64_t>(page) * block_stride;
    const int                  elements    = page_size * head_dim;
    const int                  vector_size = 8;

    for (int head = 0; head < num_heads; ++head) {
        uint16_t* head_ptr = page_ptr + static_cast<int64_t>(head) * head_stride;
        for (int index = threadIdx.x; index < elements; index += blockDim.x)
            tile[index] = head_ptr[index];
        __syncthreads();

        for (int index = threadIdx.x; index < elements; index += blockDim.x) {
            const int dim   = index / page_size;
            const int token = index % page_size;
            const int vectorized =
                (token / vector_size) * head_dim * vector_size + dim * vector_size + token % vector_size;
            if (to_vectorized)
                head_ptr[vectorized] = tile[index];
            else
                head_ptr[index] = tile[vectorized];
        }
        __syncthreads();
    }
}

at::Tensor mha_batch_prefill_graph(at::Tensor                      q,
                                   const at::Tensor&               k,
                                   at::Tensor                      v,
                                   const at::Tensor&               cu_seqlens_q,
                                   const at::Tensor&               kv_indptr,
                                   const at::Tensor&               kv_page_indices,
                                   int                             max_seqlen_q,
                                   int                             max_seqlen_k,
                                   float                           softmax_scale,
                                   at::Tensor                      out,
                                   at::Tensor                      softmax_lse,
                                   at::Tensor                      dropout_randval,
                                   at::Tensor                      rng_state,
                                   const at::Tensor&               block_table,
                                   at::Tensor                      sanitized_block_table,
                                   const at::Tensor&               seqlen_k,
                                   at::Tensor                      page_claims,
                                   bool                            linear_v,
                                   std::optional<const at::Tensor> q_descale,
                                   std::optional<const at::Tensor> k_descale,
                                   std::optional<const at::Tensor> v_descale) {
    const auto q_dtype = q.scalar_type();
    const bool is_fp8  = q_dtype == at::ScalarType::Float8_e4m3fn || q_dtype == at::ScalarType::Float8_e4m3fnuz;
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 || is_fp8,
                "graph batch prefill supports fp16, bf16, and fp8_e4m3 only");
    TORCH_CHECK(k.scalar_type() == q_dtype && v.scalar_type() == q_dtype,
                "query, key, and value must have the same dtype");
    TORCH_CHECK(k.dim() == 5 && v.dim() == 5, "graph batch prefill requires vectorized 5D K/V tensors");
    TORCH_CHECK(q.dim() == 3 && q.stride(-1) == 1, "query must be contiguous in its last dimension");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_q.is_contiguous(),
                "cu_seqlens_q must be contiguous int32");
    TORCH_CHECK(kv_indptr.scalar_type() == at::kInt && kv_indptr.is_contiguous(), "kv_indptr must be contiguous int32");
    TORCH_CHECK(kv_page_indices.scalar_type() == at::kInt && kv_page_indices.is_contiguous(),
                "kv_page_indices must be contiguous int32");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 && block_table.stride(-1) == 1,
                "block_table must be contiguous 2D int32");
    TORCH_CHECK(sanitized_block_table.scalar_type() == at::kInt && sanitized_block_table.dim() == 2
                    && sanitized_block_table.is_contiguous() && sanitized_block_table.size(0) == block_table.size(0)
                    && sanitized_block_table.size(1) >= block_table.size(1),
                "sanitized_block_table must be a sufficiently large contiguous 2D int32 tensor");
    TORCH_CHECK(seqlen_k.scalar_type() == at::kInt && seqlen_k.dim() == 1 && seqlen_k.is_contiguous(),
                "seqlen_k must be contiguous 1D int32");
    if (linear_v) {
        TORCH_CHECK(v.element_size() == sizeof(uint16_t), "linear V transpose supports fp16 and bf16 cache elements");
        TORCH_CHECK(page_claims.scalar_type() == at::kInt && page_claims.is_contiguous()
                        && page_claims.numel() >= block_table.numel(),
                    "page_claims must be contiguous int32 with one slot per block-table entry");
    }
    TORCH_CHECK(rng_state.scalar_type() == at::kLong && rng_state.numel() >= 2,
                "rng_state must contain at least two int64 values");
    TORCH_CHECK(softmax_lse.numel() == 0 && dropout_randval.numel() == 0,
                "graph batch prefill does not support LSE or dropout output");
    TORCH_CHECK(max_seqlen_k > 0, "graph batch prefill requires non-empty KV sequences");

    const int batch_size      = cu_seqlens_q.numel() - 1;
    const int num_heads       = q.size(1);
    const int head_size_q     = q.size(2);
    const int num_heads_k     = k.size(1);
    const int page_block_size = k.size(3);
    const int head_size_v     = v.size(3);
    const int num_blocks      = k.size(0);
    const int vector_size     = 16 / q.element_size();

    TORCH_CHECK(batch_size > 0 && block_table.size(0) == batch_size && seqlen_k.size(0) == batch_size,
                "graph batch prefill metadata batch size mismatch");
    TORCH_CHECK(q.size(0) == out.size(0) && out.size(1) == num_heads && out.size(2) == head_size_v
                    && out.stride(-1) == 1,
                "graph batch prefill output shape mismatch");
    TORCH_CHECK(head_size_q <= 256 && head_size_v <= 256 && head_size_q % vector_size == 0
                    && head_size_v % vector_size == 0,
                "unsupported graph batch prefill head geometry");
    TORCH_CHECK(num_heads_k > 0 && num_heads % num_heads_k == 0, "KV heads must divide query heads");
    TORCH_CHECK(page_block_size % vector_size == 0, "vectorized page size must be divisible by vector width");
    TORCH_CHECK(k.size(2) == head_size_q / vector_size && k.size(4) == vector_size, "invalid vectorized K shape");
    TORCH_CHECK(v.size(2) == page_block_size / vector_size && v.size(4) == vector_size, "invalid vectorized V shape");

    const auto expected_out_dtype = is_fp8 ? at::ScalarType::BFloat16 : q_dtype;
    TORCH_CHECK(out.scalar_type() == expected_out_dtype, "graph batch prefill output dtype mismatch");

    quant_scale_enum qscale_type = quant_scale_enum::no_scale;
    if (q_descale.has_value()) {
        TORCH_CHECK(k_descale.has_value() && v_descale.has_value(), "per-tensor scaling requires Q/K/V descales");
        qscale_type = quant_scale_enum::pertensor;
    } else {
        TORCH_CHECK(!k_descale.has_value() && !v_descale.has_value(), "K/V descales require Q descale");
    }

    std::string dtype_str = torchDTypeToStr(c10::scalarTypeToTypeMeta(q_dtype));
    if (is_fp8)
        dtype_str = "fp8bf16";

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q.device()};
    const hipStream_t                                 stream      = at::hip::getCurrentHIPStream();
    constexpr int                                     threads     = 256;
    const int                                         output_cols = sanitized_block_table.size(1);
    const dim3                                        grid((output_cols + threads - 1) / threads, batch_size);

    auto transpose_linear_v = [&](bool to_vectorized) {
        if (!linear_v)
            return;
        hipMemsetAsync(page_claims.data_ptr<int32_t>(), 0xff, page_claims.numel() * sizeof(int32_t), stream);
        const int    entries      = batch_size * block_table.size(1);
        const size_t shared_bytes = page_block_size * head_size_v * sizeof(uint16_t);
        hipLaunchKernelGGL(transpose_linear_v_pages_kernel,
                           dim3(entries),
                           dim3(threads),
                           shared_bytes,
                           stream,
                           reinterpret_cast<uint16_t*>(v.data_ptr()),
                           block_table.data_ptr<int32_t>(),
                           seqlen_k.data_ptr<int32_t>(),
                           page_claims.data_ptr<int32_t>(),
                           batch_size,
                           block_table.size(1),
                           num_blocks,
                           num_heads_k,
                           page_block_size,
                           head_size_v,
                           v.stride(0),
                           v.stride(1),
                           to_vectorized);
    };

    transpose_linear_v(true);
    hipLaunchKernelGGL(sanitize_block_table_kernel,
                       grid,
                       dim3(threads),
                       0,
                       stream,
                       block_table.data_ptr<int32_t>(),
                       seqlen_k.data_ptr<int32_t>(),
                       sanitized_block_table.data_ptr<int32_t>(),
                       block_table.size(1),
                       output_cols,
                       page_block_size);

    std::optional<const at::Tensor> bias;
    std::optional<const at::Tensor> alibi_slopes;
    std::optional<const at::Tensor> kv_block_descale;
    std::optional<const at::Tensor> kv_last_page_lens;
    mask_info                       mask{};
    mask.type           = mask_enum::mask_bottom_right;
    mask.seqlen_q       = max_seqlen_q;
    mask.seqlen_k       = max_seqlen_k;
    mask.y              = max_seqlen_q;
    mask.x              = max_seqlen_k - max_seqlen_q + 1;
    mask.left           = -1;
    mask.right          = 0;
    mask.sink           = 0;
    auto* rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
    auto  args          = get_ck_fmha_batch_prefill_args(false,
                                               false,
                                               mask,
                                               batch_size,
                                               max_seqlen_q,
                                               num_heads,
                                               num_heads_k,
                                               head_size_q,
                                               head_size_v,
                                               num_blocks,
                                               page_block_size,
                                               ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT,
                                               q,
                                               k,
                                               v,
                                               cu_seqlens_q,
                                               kv_indptr,
                                               kv_page_indices,
                                               std::nullopt,
                                               bias,
                                               alibi_slopes,
                                               q_descale,
                                               k_descale,
                                               v_descale,
                                               kv_block_descale,
                                               out,
                                               softmax_lse,
                                               dropout_randval,
                                               softmax_scale,
                                               0.0f,
                                               0.0f,
                                                         {rng_state_ptr, rng_state_ptr + 1},
                                               kv_last_page_lens);

    args.num_total_pages          = num_blocks;
    args.page_block_size          = page_block_size;
    args.kv_memory_layout         = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    args.kv_lookup_table          = ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;
    args.kv_indptr                = nullptr;
    args.kv_page_indices          = sanitized_block_table.data_ptr();
    args.kv_last_page_lens        = nullptr;
    args.seqlen_k_ptr             = seqlen_k.data_ptr();
    args.batch_stride_block_table = sanitized_block_table.stride(0);

    ck_tile::stream_config stream_config{stream};
    const float            elapsed = aiter::mha_batch_prefill(
        args, stream_config, dtype_str, true, mask.type, bias_enum::no_bias, false, qscale_type, false);
    transpose_linear_v(false);
    TORCH_CHECK(elapsed >= 0, "no matching graph batch prefill kernel found");
    return out;
}

}  // namespace aiter::torch_itfs

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_batch_prefill_graph",
          &aiter::torch_itfs::mha_batch_prefill_graph,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("cu_seqlens_q"),
          py::arg("kv_indptr"),
          py::arg("kv_page_indices"),
          py::arg("max_seqlen_q"),
          py::arg("max_seqlen_k"),
          py::arg("softmax_scale"),
          py::arg("out"),
          py::arg("softmax_lse"),
          py::arg("dropout_randval"),
          py::arg("rng_state"),
          py::arg("block_table"),
          py::arg("sanitized_block_table"),
          py::arg("seqlen_k"),
          py::arg("page_claims"),
          py::arg("linear_v"),
          py::arg("q_descale") = std::nullopt,
          py::arg("k_descale") = std::nullopt,
          py::arg("v_descale") = std::nullopt);
}
