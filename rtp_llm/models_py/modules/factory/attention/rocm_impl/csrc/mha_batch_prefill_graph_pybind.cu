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

__global__ void copy_linear_kv_pages_kernel(const uint16_t* k,
                                            const uint16_t* v,
                                            uint16_t*       k_scratch,
                                            uint16_t*       v_scratch,
                                            const int32_t*  sanitized_block_table,
                                            int32_t*        scratch_block_table,
                                            int             entries,
                                            int             num_blocks,
                                            int             num_heads,
                                            int             page_size,
                                            int             head_size_k,
                                            int             head_size_v,
                                            int64_t         k_block_stride,
                                            int64_t         k_head_stride,
                                            int64_t         v_block_stride,
                                            int64_t         v_head_stride,
                                            int64_t         scratch_k_block_stride,
                                            int64_t         scratch_k_head_stride,
                                            int64_t         scratch_v_block_stride,
                                            int64_t         scratch_v_head_stride) {
    const int entry = blockIdx.x;
    const int head  = blockIdx.y;
    if (entry >= entries || head >= num_heads)
        return;

    const int32_t page       = sanitized_block_table[entry];
    const bool    valid_page = page >= 0 && page < num_blocks;

    // Every table slot owns a distinct scratch page. This keeps duplicate/shared
    // live pages race-free and gives CK an in-range page for malformed entries.
    if (head == 0 && threadIdx.x == 0)
        scratch_block_table[entry] = entry;

    uint16_t* k_dst = k_scratch + static_cast<int64_t>(entry) * scratch_k_block_stride
                      + static_cast<int64_t>(head) * scratch_k_head_stride;
    uint16_t* v_dst = v_scratch + static_cast<int64_t>(entry) * scratch_v_block_stride
                      + static_cast<int64_t>(head) * scratch_v_head_stride;

    const uint16_t* k_src = nullptr;
    const uint16_t* v_src = nullptr;
    if (valid_page) {
        k_src = k + static_cast<int64_t>(page) * k_block_stride + static_cast<int64_t>(head) * k_head_stride;
        v_src = v + static_cast<int64_t>(page) * v_block_stride + static_cast<int64_t>(head) * v_head_stride;
    }

    const int k_elements = page_size * head_size_k;
    for (int index = threadIdx.x; index < k_elements; index += blockDim.x)
        k_dst[index] = valid_page ? k_src[index] : uint16_t{0};

    // The non-ASM writer stores V logically as [head_size_v, page_size], even
    // though Python exposes the same bytes through a 5D placeholder view.
    // Convert those bytes to CK's [page_size/vector, head_size_v, vector] order.
    constexpr int vector_size = 8;
    const int     v_elements  = page_size * head_size_v;
    for (int index = threadIdx.x; index < v_elements; index += blockDim.x) {
        const int dim   = index / page_size;
        const int token = index % page_size;
        const int v_offset =
            (token / vector_size) * head_size_v * vector_size + dim * vector_size + token % vector_size;
        v_dst[v_offset] = valid_page ? v_src[index] : uint16_t{0};
    }
}

at::Tensor mha_batch_prefill_graph(at::Tensor                      q,
                                   const at::Tensor&               k,
                                   const at::Tensor&               v,
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
                                   at::Tensor                      scratch_block_table,
                                   bool                            linear_v,
                                   std::optional<const at::Tensor> q_descale,
                                   std::optional<const at::Tensor> k_descale,
                                   std::optional<const at::Tensor> v_descale,
                                   std::optional<const at::Tensor> k_scratch,
                                   std::optional<const at::Tensor> v_scratch) {
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
    TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 && block_table.is_contiguous()
                    && block_table.size(1) > 0,
                "block_table must be non-empty contiguous 2D int32");
    TORCH_CHECK(sanitized_block_table.scalar_type() == at::kInt && sanitized_block_table.dim() == 2
                    && sanitized_block_table.is_contiguous() && sanitized_block_table.size(0) == block_table.size(0)
                    && sanitized_block_table.size(1) >= block_table.size(1),
                "sanitized_block_table must be a sufficiently large contiguous 2D int32 tensor");
    TORCH_CHECK(seqlen_k.scalar_type() == at::kInt && seqlen_k.dim() == 1 && seqlen_k.is_contiguous(),
                "seqlen_k must be contiguous 1D int32");
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
    const int vector_size     = 16 / k.element_size();
    const int head_size_k     = k.size(2) * vector_size;

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
    TORCH_CHECK(k.size(0) == v.size(0) && k.size(1) == v.size(1), "K/V page and head counts must match");
    TORCH_CHECK(head_size_k == head_size_q && k.size(4) == vector_size, "invalid vectorized K shape");
    TORCH_CHECK(v.size(2) == page_block_size / vector_size && v.size(4) == vector_size, "invalid vectorized V shape");
    TORCH_CHECK(k.stride(4) == 1 && k.stride(3) == vector_size && k.stride(2) == page_block_size * vector_size
                    && k.stride(1) == static_cast<int64_t>(head_size_k) * page_block_size,
                "K must use contiguous vectorized per-head layout");
    TORCH_CHECK(v.stride(4) == 1 && v.stride(3) == vector_size
                    && v.stride(2) == static_cast<int64_t>(head_size_v) * vector_size
                    && v.stride(1) == static_cast<int64_t>(head_size_v) * page_block_size,
                "V must use contiguous per-head storage");

    if (linear_v) {
        TORCH_CHECK(k.element_size() == sizeof(uint16_t) && v.element_size() == sizeof(uint16_t),
                    "linear V scratch conversion supports fp16 and bf16 cache elements");
        TORCH_CHECK(scratch_block_table.scalar_type() == at::kInt && scratch_block_table.is_contiguous()
                        && scratch_block_table.sizes() == sanitized_block_table.sizes(),
                    "scratch_block_table must match sanitized_block_table");
        TORCH_CHECK(k_scratch.has_value() && v_scratch.has_value(), "linear V requires K/V scratch tensors");
        TORCH_CHECK(k_scratch->scalar_type() == k.scalar_type() && v_scratch->scalar_type() == v.scalar_type(),
                    "K/V scratch dtype must match the corresponding cache dtype");
        TORCH_CHECK(k_scratch->is_contiguous() && v_scratch->is_contiguous(), "K/V scratch must be contiguous");
        TORCH_CHECK(k_scratch->device() == k.device() && v_scratch->device() == v.device()
                        && scratch_block_table.device() == block_table.device(),
                    "linear V scratch tensors must be on the cache device");
        TORCH_CHECK(k_scratch->data_ptr() != k.data_ptr() && v_scratch->data_ptr() != v.data_ptr(),
                    "linear V K/V scratch must not alias the live cache");
        TORCH_CHECK(scratch_block_table.data_ptr() != sanitized_block_table.data_ptr(),
                    "linear V scratch block table must not alias sanitized_block_table");
    }

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

    at::Tensor kernel_k           = k;
    at::Tensor kernel_v           = v;
    at::Tensor kernel_block_table = sanitized_block_table;
    int        kernel_num_blocks  = num_blocks;
    if (linear_v) {
        const at::Tensor& scratch_k = *k_scratch;
        const at::Tensor& scratch_v = *v_scratch;
        const int         entries   = sanitized_block_table.numel();
        TORCH_CHECK(scratch_k.dim() == 5 && scratch_k.size(0) == entries && scratch_k.size(1) == num_heads_k
                        && scratch_k.size(2) == head_size_q / vector_size && scratch_k.size(3) == page_block_size
                        && scratch_k.size(4) == vector_size,
                    "invalid K scratch shape");
        TORCH_CHECK(scratch_v.dim() == 5 && scratch_v.size(0) == entries && scratch_v.size(1) == num_heads_k
                        && scratch_v.size(2) == page_block_size / vector_size && scratch_v.size(3) == head_size_v
                        && scratch_v.size(4) == vector_size,
                    "invalid V scratch shape");
        hipLaunchKernelGGL(copy_linear_kv_pages_kernel,
                           dim3(entries, num_heads_k),
                           dim3(threads),
                           0,
                           stream,
                           reinterpret_cast<const uint16_t*>(k.data_ptr()),
                           reinterpret_cast<const uint16_t*>(v.data_ptr()),
                           reinterpret_cast<uint16_t*>(scratch_k.data_ptr()),
                           reinterpret_cast<uint16_t*>(scratch_v.data_ptr()),
                           sanitized_block_table.data_ptr<int32_t>(),
                           scratch_block_table.data_ptr<int32_t>(),
                           entries,
                           num_blocks,
                           num_heads_k,
                           page_block_size,
                           head_size_k,
                           head_size_v,
                           k.stride(0),
                           k.stride(1),
                           v.stride(0),
                           v.stride(1),
                           scratch_k.stride(0),
                           scratch_k.stride(1),
                           scratch_v.stride(0),
                           scratch_v.stride(1));
        kernel_k           = scratch_k;
        kernel_v           = scratch_v;
        kernel_block_table = scratch_block_table;
        kernel_num_blocks  = entries;
    }

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
                                               kernel_num_blocks,
                                               page_block_size,
                                               ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT,
                                               q,
                                               kernel_k,
                                               kernel_v,
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

    args.num_total_pages          = kernel_num_blocks;
    args.page_block_size          = page_block_size;
    args.kv_memory_layout         = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    args.kv_lookup_table          = ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;
    args.kv_indptr                = nullptr;
    args.kv_page_indices          = kernel_block_table.data_ptr();
    args.kv_last_page_lens        = nullptr;
    args.seqlen_k_ptr             = seqlen_k.data_ptr();
    args.batch_stride_block_table = kernel_block_table.stride(0);

    ck_tile::stream_config stream_config{stream};
    const float            elapsed = aiter::mha_batch_prefill(
        args, stream_config, dtype_str, true, mask.type, bias_enum::no_bias, false, qscale_type, false);
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
          py::arg("scratch_block_table"),
          py::arg("linear_v"),
          py::arg("q_descale") = std::nullopt,
          py::arg("k_descale") = std::nullopt,
          py::arg("v_descale") = std::nullopt,
          py::arg("k_scratch") = std::nullopt,
          py::arg("v_scratch") = std::nullopt);
}
