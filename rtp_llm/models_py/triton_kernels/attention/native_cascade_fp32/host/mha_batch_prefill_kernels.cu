// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "mha_common.h"
#include "mha_fwd.h"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

namespace aiter {
namespace torch_itfs {
fmha_batch_prefill_args
get_ck_fmha_batch_prefill_args(bool has_lse,
                               bool has_dropout_randval,
                               const mask_info& mask,
                               // sizes
                               const int b,
                               const int max_seqlen_q,
                               const int h,
                               const int h_k,
                               const int d,
                               const int d_v,
                               const int num_total_pages,
                               const int page_block_size,
                               ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout,
                               // device pointers
                               const at::Tensor q,
                               const at::Tensor k,
                               const at::Tensor v,
                               const at::Tensor seqlens_q,
                               const at::Tensor kv_indptr,
                               const at::Tensor kv_page_indices,
                               std::optional<const at::Tensor> sink_ptr_,
                               std::optional<const at::Tensor>& bias_,
                               std::optional<const at::Tensor>& alibi_slopes_,
                               // Per-tensor descale for PERTENSOR mode (Q/K/V each have one scale value)
                               std::optional<const at::Tensor>& q_descale,   // [1] per-tensor Q descale
                               std::optional<const at::Tensor>& k_descale,   // [1] per-tensor K descale
                               std::optional<const at::Tensor>& v_descale,   // [1] per-tensor V descale
                               // Per-page descale for KV_BLOCKSCALE mode (Q per-tensor, K/V per-page)
                               // Mutually exclusive with k_descale/v_descale
                               std::optional<const at::Tensor>& kv_block_descale, // [num_block, num_kv_head, 2]
                               at::Tensor out,
                               at::Tensor softmax_lse,
                               at::Tensor dropout_randval,
                               float softmax_scale,
                               float logits_soft_cap,
                               float p_dropout,
                               std::pair<uint64_t*, uint64_t*> drop_seed_offset,
                               std::optional<const at::Tensor>& kv_last_page_lens_)
{
    // q: (total_q, nheads, d)
    // o: (total_q, nheads, d_v)

    // bias:(total_q, max_seqlen_k)
    // alibi_slopes:(batch, nheads) or (nhead)
    // lse: (nheads, total_q)
    // randval: (nheads, total_q, max_seqlen_k)

    ck_tile::index_t total_q = q.size(0);
    ck_tile::index_t total_k = k.size(0) * page_block_size;

    ck_tile::index_t stride_q       = q.stride(-3);
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;

    const int k_vector_size = 16 / static_cast<int>(k.element_size());

    if(kv_memory_layout == ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
    {
        TORCH_CHECK(
            k.dim() == 5,
            "K tensor must be 5D [NumBlocks, NumHeads, HeadDim/kVectorSize, PageSize, kVectorSize]");
        TORCH_CHECK(
            v.dim() == 5,
            "V tensor must be 5D [NumBlocks, NumHeads, PageSize/kVectorSize, HeadDim, kVectorSize]");

        // Vectorized layout strides
        // K: [NumBlocks, NumHeads, HeadDim/kVectorSize, PageSize, kVectorSize] -> stride(-2) is PageSize
        // V: [NumBlocks, NumHeads, PageSize/kVectorSize, HeadDim, kVectorSize] -> stride(-2) is HeadDim
        stride_k = k.stride(-2);
        stride_v = v.stride(-2);

        const int64_t k_stride_batch = k.stride(0);
        const int64_t k_stride_head  = k.stride(1);
        const int64_t k_stride_dvec  = k.stride(2);
        const int64_t k_stride_tok   = k.stride(3);
        const int64_t k_stride_vec   = k.stride(4);

        TORCH_CHECK(stride_k == k_vector_size,
                    "stride_k (PageSize stride) must be ",
                    k_vector_size,
                    " in 5D vectorized layout");
        TORCH_CHECK(stride_v == k_vector_size,
                    "stride_v (HeadDim stride) must be ",
                    k_vector_size,
                    " in 5D vectorized layout");
        TORCH_CHECK(k_stride_vec == 1 && k.size(-1) == k_vector_size,
                    "K last dim must be ",
                    k_vector_size,
                    " and contiguous");
        TORCH_CHECK(k_stride_tok == k_vector_size,
                    "K page stride must be ",
                    k_vector_size,
                    " in 5D vectorized layout");
        TORCH_CHECK(k_stride_dvec == static_cast<int64_t>(page_block_size) * k_vector_size,
                    "K head-dim stride must be page_size * vector_size");
        TORCH_CHECK(k_stride_head >= static_cast<int64_t>(d) * page_block_size,
                    "K head stride must be >= head_dim * page_size");
        TORCH_CHECK(k_stride_batch >= static_cast<int64_t>(h_k) * k_stride_head,
                    "K batch stride must be >= num_heads * head_stride");
        TORCH_CHECK(k_stride_head % k_vector_size == 0,
                    "K head stride must be a multiple of vector size");
        TORCH_CHECK(k_stride_batch % k_vector_size == 0,
                    "K batch stride must be a multiple of vector size");

        const int64_t v_stride_batch = v.stride(0);
        const int64_t v_stride_head  = v.stride(1);
        const int64_t v_stride_tok   = v.stride(2);
        const int64_t v_stride_dim   = v.stride(3);
        const int64_t v_stride_vec   = v.stride(4);

        TORCH_CHECK(v_stride_vec == 1 && v.size(-1) == k_vector_size,
                    "V last dim must be ",
                    k_vector_size,
                    " and contiguous");
        TORCH_CHECK(v_stride_dim == k_vector_size,
                    "V head-dim stride must be ",
                    k_vector_size,
                    " in 5D vectorized layout");
        TORCH_CHECK(v_stride_tok == static_cast<int64_t>(d_v) * k_vector_size,
                    "V page stride must be head_dim * vector_size");
        TORCH_CHECK(v_stride_head >= static_cast<int64_t>(d_v) * page_block_size,
                    "V head stride must be >= head_dim * page_size");
        TORCH_CHECK(v_stride_batch >= static_cast<int64_t>(h_k) * v_stride_head,
                    "V batch stride must be >= num_heads * head_stride");
        TORCH_CHECK(v_stride_head % k_vector_size == 0,
                    "V head stride must be a multiple of vector size");
        TORCH_CHECK(v_stride_batch % k_vector_size == 0,
                    "V batch stride must be a multiple of vector size");
    }
    else
    {
        if(k.dim() == 4)
        {
            TORCH_CHECK(v.dim() == 4,
                        "V tensor must be 4D [NumBlocks, PageSize, NumHeads, HeadDim]");

            // Linear layout strides
            // K/V: [NumBlocks, PageSize, NumHeads, HeadDim] -> stride(1) is PageSize
            stride_k = k.stride(1);
            stride_v = v.stride(1);

            const int64_t k_stride_batch = k.stride(0);
            const int64_t k_stride_page  = k.stride(1);
            const int64_t k_stride_head  = k.stride(2);
            const int64_t k_stride_dim   = k.stride(3);

            TORCH_CHECK(k_stride_dim == 1, "K last dim must be contiguous");
            TORCH_CHECK(k_stride_head >= d, "K head stride must be >= head_dim");
            TORCH_CHECK(k_stride_page >= static_cast<int64_t>(h_k) * k_stride_head,
                        "K page stride must be >= num_heads * head_stride");
            TORCH_CHECK(k_stride_batch >= static_cast<int64_t>(page_block_size) * k_stride_page,
                        "K batch stride must be >= page_size * page_stride");
            TORCH_CHECK(k_stride_head % k_vector_size == 0,
                        "K head stride must be a multiple of vector size");
            TORCH_CHECK(k_stride_page % k_vector_size == 0,
                        "K page stride must be a multiple of vector size");
            TORCH_CHECK(k_stride_batch % k_vector_size == 0,
                        "K batch stride must be a multiple of vector size");

            const int64_t v_stride_batch = v.stride(0);
            const int64_t v_stride_page  = v.stride(1);
            const int64_t v_stride_head  = v.stride(2);
            const int64_t v_stride_dim   = v.stride(3);

            TORCH_CHECK(v_stride_dim == 1, "V last dim must be contiguous");
            TORCH_CHECK(v_stride_head >= d_v, "V head stride must be >= head_dim");
            TORCH_CHECK(v_stride_page >= static_cast<int64_t>(h_k) * v_stride_head,
                        "V page stride must be >= num_heads * head_stride");
            TORCH_CHECK(v_stride_batch >= static_cast<int64_t>(page_block_size) * v_stride_page,
                        "V batch stride must be >= page_size * page_stride");
            TORCH_CHECK(v_stride_head % k_vector_size == 0,
                        "V head stride must be a multiple of vector size");
            TORCH_CHECK(v_stride_page % k_vector_size == 0,
                        "V page stride must be a multiple of vector size");
            TORCH_CHECK(v_stride_batch % k_vector_size == 0,
                        "V batch stride must be a multiple of vector size");
        }
        else if(k.dim() == 3)
        {
            TORCH_CHECK(page_block_size == 1,
                        "3D K/V tensors require page_block_size == 1");
            TORCH_CHECK(v.dim() == 3,
                        "V tensor must be 3D [NumBlocks, NumHeads, HeadDim]");

            // Treat 3D K/V as PageSize=1 linear layout.
            stride_k = k.stride(1);
            stride_v = v.stride(1);

            const int64_t k_stride_batch = k.stride(0);
            const int64_t k_stride_head  = k.stride(1);
            const int64_t k_stride_dim   = k.stride(2);

            TORCH_CHECK(k_stride_dim == 1, "K last dim must be contiguous");
            TORCH_CHECK(k_stride_head >= d, "K head stride must be >= head_dim");
            TORCH_CHECK(k_stride_batch >= static_cast<int64_t>(h_k) * k_stride_head,
                        "K batch stride must be >= num_heads * head_stride");
            TORCH_CHECK(k_stride_head % k_vector_size == 0,
                        "K head stride must be a multiple of vector size");
            TORCH_CHECK(k_stride_batch % k_vector_size == 0,
                        "K batch stride must be a multiple of vector size");

            const int64_t v_stride_batch = v.stride(0);
            const int64_t v_stride_head  = v.stride(1);
            const int64_t v_stride_dim   = v.stride(2);

            TORCH_CHECK(v_stride_dim == 1, "V last dim must be contiguous");
            TORCH_CHECK(v_stride_head >= d_v, "V head stride must be >= head_dim");
            TORCH_CHECK(v_stride_batch >= static_cast<int64_t>(h_k) * v_stride_head,
                        "V batch stride must be >= num_heads * head_stride");
            TORCH_CHECK(v_stride_head % k_vector_size == 0,
                        "V head stride must be a multiple of vector size");
            TORCH_CHECK(v_stride_batch % k_vector_size == 0,
                        "V batch stride must be a multiple of vector size");
        }
        else
        {
            TORCH_CHECK(false,
                        "K tensor must be 4D [NumBlocks, PageSize, NumHeads, HeadDim] or "
                        "3D [NumBlocks, NumHeads, HeadDim] (page_block_size == 1)");
        }
    }

    ck_tile::index_t stride_o       = out.stride(-3);
    ck_tile::index_t stride_randval = has_dropout_randval ? dropout_randval.stride(1) : 0;

    ck_tile::index_t nhead_stride_q       = q.stride(-2);
    const bool is_vectorized_layout =
        kv_memory_layout == ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
    // Vectorized: head dim at index 1. Linear: head dim at index 2.
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    if(is_vectorized_layout)
    {
        nhead_stride_k = k.stride(1);
        nhead_stride_v = v.stride(1);
    }
    else if(k.dim() == 3)
    {
        nhead_stride_k = k.stride(1);
        nhead_stride_v = v.stride(1);
    }
    else
    {
        nhead_stride_k = k.stride(2);
        nhead_stride_v = v.stride(2);
    }
    
    ck_tile::index_t nhead_stride_o       = out.stride(-2);
    ck_tile::index_t nhead_stride_lse     = has_lse ? softmax_lse.stride(0) : 0;
    ck_tile::index_t nhead_stride_randval = has_dropout_randval ? dropout_randval.stride(0) : 0;

    ck_tile::index_t batch_stride_q       = 0;
    ck_tile::index_t batch_stride_k       = k.stride(0);
    ck_tile::index_t batch_stride_v       = v.stride(0);
    ck_tile::index_t batch_stride_o       = 0;
    ck_tile::index_t batch_stride_lse     = 0;
    ck_tile::index_t batch_stride_randval = 0;

    void* bias_ptr                     = nullptr;
    ck_tile::index_t stride_bias       = 0;
    ck_tile::index_t nhead_stride_bias = 0;
    ck_tile::index_t batch_stride_bias = 0;

    if(bias_.has_value())
    {
        auto bias = bias_.value();
        CHECK_DEVICE(bias);
        TORCH_CHECK(bias.stride(-1) == 1, "bias tensor must have contiguous last dimension");
        TORCH_CHECK(bias.dim() == 2, "only support 2d bias");
        bias_ptr = bias.data_ptr();
        if(bias.dim() == 2)
            stride_bias = bias.stride(0);
    }
    else if(alibi_slopes_.has_value())
    {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1,
                    "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) ||
                    alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        bias_ptr    = alibi_slopes.data_ptr();
        stride_bias = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    void* kv_last_page_lens_ptr = nullptr;
    if(kv_last_page_lens_.has_value())
    {
        auto kv_last_page_lens = kv_last_page_lens_.value();
        CHECK_DEVICE(kv_last_page_lens);
        TORCH_CHECK(kv_last_page_lens.dim() == 1, "kv_last_page_lens must be 1d");
        kv_last_page_lens_ptr = kv_last_page_lens.data_ptr();
    }

    fmha_batch_prefill_args args{};  // zero-initialize all fields

    args.q_ptr             = q.data_ptr();
    args.k_ptr             = k.data_ptr();
    args.v_ptr             = v.data_ptr();
    args.q_descale_ptr     = q_descale.has_value() ? q_descale.value().data_ptr() : nullptr;
    args.k_descale_ptr     = k_descale.has_value() ? k_descale.value().data_ptr() : nullptr;
    args.v_descale_ptr     = v_descale.has_value() ? v_descale.value().data_ptr() : nullptr;
    // sink_ptr is independent of sink_size: when provided, the kernel always reads
    // it as per-head logit values for the virtual sink token (sink_value = *ptr / scale_s).
    // When null, sink_value defaults to -inf (virtual token excluded from softmax).
    args.sink_ptr          = sink_ptr_.has_value() ? sink_ptr_.value().data_ptr() : nullptr;
    args.bias_ptr          = bias_ptr;
    args.rand_val_ptr      = has_dropout_randval ? dropout_randval.data_ptr() : nullptr;
    args.lse_ptr           = has_lse ? softmax_lse.data_ptr() : nullptr;
    args.o_ptr             = out.data_ptr();
    args.seqstart_q_ptr    = seqlens_q.data_ptr();
    args.seqlen_q          = total_q;
    args.seqlen_k          = total_k;
    args.batch             = b;
    args.max_seqlen_q      = max_seqlen_q;
    args.hdim_q            = d;
    args.hdim_v            = d_v;
    args.nhead_q           = h;
    args.nhead_k           = h_k;
    args.num_total_pages   = num_total_pages;
    args.page_block_size   = page_block_size;
    args.kv_memory_layout  = kv_memory_layout;
    args.kv_lookup_table   = ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D;
    args.kv_indptr         = kv_indptr.data_ptr();
    args.kv_page_indices   = kv_page_indices.data_ptr();
    args.kv_last_page_lens = kv_last_page_lens_ptr;
    args.seqlen_k_ptr      = nullptr;
    args.batch_stride_block_table = 0;
    args.scale_s           = softmax_scale;
    args.scale_p           = 1;
    args.scale_o           = 1;

    args.logits_soft_cap = logits_soft_cap;

    args.stride_q             = stride_q;
    args.stride_k             = stride_k;
    args.stride_v             = stride_v;
    args.stride_bias          = stride_bias;
    args.stride_randval       = stride_randval;
    args.stride_o             = stride_o;
    args.nhead_stride_q       = nhead_stride_q;
    args.nhead_stride_k       = nhead_stride_k;
    args.nhead_stride_v       = nhead_stride_v;
    args.nhead_stride_bias    = nhead_stride_bias;
    args.nhead_stride_randval = nhead_stride_randval;
    args.nhead_stride_lse     = nhead_stride_lse;
    args.nhead_stride_o       = nhead_stride_o;
    args.batch_stride_q       = batch_stride_q;
    args.batch_stride_k       = batch_stride_k;
    args.batch_stride_v       = batch_stride_v;
    args.batch_stride_bias    = batch_stride_bias;
    args.batch_stride_randval = batch_stride_randval;
    args.batch_stride_lse     = batch_stride_lse;
    args.batch_stride_o       = batch_stride_o;
    args.window_size_left     = mask.left;
    args.window_size_right    = mask.right;
    args.sink_size            = mask.sink;
    args.mask_type            = static_cast<ck_tile::index_t>(mask.type);
    args.p_drop               = p_dropout;
    args.s_randval            = has_dropout_randval;
    args.drop_seed_offset     = drop_seed_offset;

    // KV_BLOCKSCALE: per-page K/V descales (Q per-tensor, K/V per-page)
    // kv_block_descale layout: [num_block, num_kv_head, 2] where 2 = (k_descale, v_descale)
    if(kv_block_descale.has_value())
    {
        auto kv_block_descale_tensor = kv_block_descale.value();
        CHECK_DEVICE(kv_block_descale_tensor);
        TORCH_CHECK(kv_block_descale_tensor.scalar_type() == at::kFloat,
                    "kv_block_descale must be float32");
        TORCH_CHECK(kv_block_descale_tensor.dim() == 3,
                    "kv_block_descale must be 3D [num_block, num_kv_head, 2]");
        TORCH_CHECK(kv_block_descale_tensor.size(0) == num_total_pages,
                    "kv_block_descale first dim must match num_total_pages");
        TORCH_CHECK(kv_block_descale_tensor.size(1) == h_k,
                    "kv_block_descale second dim must match num_kv_heads");
        TORCH_CHECK(kv_block_descale_tensor.size(2) == 2,
                    "kv_block_descale third dim must be 2 (k_scale, v_scale)");

        // Split into separate K and V descale pointers
        // k_descale: [num_block, num_kv_head] at kv_block_descale[..., 0]
        // v_descale: [num_block, num_kv_head] at kv_block_descale[..., 1]
        auto k_descale_view = kv_block_descale_tensor.select(-1, 0);
        auto v_descale_view = kv_block_descale_tensor.select(-1, 1);

        args.k_descale_ptr                  = k_descale_view.data_ptr();
        args.v_descale_ptr                  = v_descale_view.data_ptr();
        args.nblock_stride_kv_block_descale = k_descale_view.stride(0);
        args.nhead_stride_kv_block_descale  = k_descale_view.stride(1);
    }

    return args;
}

std::vector<at::Tensor>
mha_batch_prefill(at::Tensor& q,       // [total_q, hq, d]
                  const at::Tensor& k, // [num_blocks, hk, d/k_vector_size, block_size, k_vector_size]
                  const at::Tensor& v, // [num_blocks, hk, block_size/k_vector_size, d, k_vector_size]
                  const at::Tensor& cu_seqlens_q, // [b+1]
                  const at::Tensor& kv_indptr,    // [b+1]
                  const at::Tensor& kv_page_indices,
                  int max_seqlen_q,
                  int max_seqlen_k,
                  float p_dropout,
                  float softmax_scale,
                  float logits_soft_cap,
                  bool zero_tensors,
                  bool is_causal,
                  int window_size_left,
                  int window_size_right,
                  int sink_size,
                  bool return_softmax_lse,
                  bool return_dropout_randval,
                  std::optional<at::Tensor> out_,                // [total_q, hq, d]
                  std::optional<const at::Tensor> bias_,         // [total_q, max_seqlen_k]
                  std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
                  std::optional<const at::Tensor> q_descale,     // [1]
                  std::optional<const at::Tensor> k_descale,     // [1]
                  std::optional<const at::Tensor> v_descale,     // [1]
                  std::optional<const at::Tensor> kv_block_descale,      // [num_block, num_kv_head, 2] for KV_BLOCKSCALE
                  std::optional<const at::Tensor> kv_last_page_lens_,
                  std::optional<const at::Tensor> block_table_,
                  std::optional<const at::Tensor> seqlen_k_,
                  std::optional<const at::Tensor> sink_ptr,      // [hq]
                  std::optional<at::Generator> gen_
                )
{
    auto q_dtype = q.scalar_type();
    bool is_qkv_fp8 =
        q_dtype == at::ScalarType::Float8_e4m3fn || q_dtype == at::ScalarType::Float8_e4m3fnuz;
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 ||
                    is_qkv_fp8,
                "FlashAttention only support fp16, bf16 and fp8_e4m3 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(kv_indptr.dtype() == torch::kInt32, "kv_indptr must have dtype int32");

    std::string dtype_str = torchDTypeToStr(c10::scalarTypeToTypeMeta(q_dtype));
    if(is_qkv_fp8)
    {
        if(!out_.has_value() || out_.value().dtype() == at::ScalarType::BFloat16)
            dtype_str = "fp8bf16"; // BF16 output is required for FP8 input due to current kernel
                                   // implementation constraints
        else
            TORCH_CHECK(false, "For FP8 input, output must have dtype BF16 for now");
    }

    // Validate descale tensor combinations:
    // - PERTENSOR mode: q_descale, k_descale, v_descale all provided
    // - KV_BLOCKSCALE mode: q_descale + kv_block_descale provided, k_descale/v_descale NOT provided
    // - NO_SCALE mode: none provided
    quant_scale_enum qscale_type;
    if(kv_block_descale.has_value())
    {
        // KV_BLOCKSCALE: Q per-tensor, K/V per-page
        TORCH_CHECK(q_descale.has_value(),
                    "kv_block_descale requires q_descale for per-tensor Q scaling");
        TORCH_CHECK(!k_descale.has_value() && !v_descale.has_value(),
                    "kv_block_descale and k_descale/v_descale are mutually exclusive. "
                    "Use kv_block_descale for per-page KV scaling, or k_descale/v_descale for per-tensor scaling.");
        qscale_type = quant_scale_enum::kv_blockscale;
    }
    else if(q_descale.has_value())
    {
        // PERTENSOR mode: all three per-tensor descales required
        TORCH_CHECK(k_descale.has_value() && v_descale.has_value(),
                    "For per-tensor mode, q_descale, k_descale, v_descale must all be provided");
        qscale_type = quant_scale_enum::pertensor;
    }
    else
    {
        // NO_SCALE mode: none should be provided
        TORCH_CHECK(!k_descale.has_value() && !v_descale.has_value(),
                    "k_descale and v_descale require q_descale to also be provided");
        qscale_type = quant_scale_enum::no_scale;
    }

    const int k_vector_size = 16 / static_cast<int>(q.element_size());

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(kv_indptr);

    CHECK_DEVICE(kv_page_indices);
    TORCH_CHECK(kv_page_indices.dtype() == torch::kInt32,
                "kv_page_indices must have dtype torch.int32");
    TORCH_CHECK(kv_page_indices.stride(-1) == 1,
                "kv_page_indices must have contiguous last dimension");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(kv_indptr);

    const auto sizes = q.sizes();

    const int batch_size  = cu_seqlens_q.numel() - 1;
    int num_heads         = sizes[1];
    const int head_size_q = sizes[2];
    
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout;
    int num_heads_k     = 0;
    int page_block_size = 0;
    int head_size_v     = 0;
    int num_blocks      = 0;

    if(k.dim() == 5)
    {
        kv_memory_layout = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT;
        TORCH_CHECK(
            v.dim() == 5,
            "V tensor must be 5D [NumBlocks, NumHeads, PageSize/kVectorSize, HeadDim, kVectorSize]");

        // K: [NumBlocks, NumHeads, HeadDim/kVectorSize, PageSize, kVectorSize]
        num_heads_k     = k.size(1);
        page_block_size = k.size(3);
        TORCH_CHECK(page_block_size % k_vector_size == 0,
                    "Vectorized KV requires page size divisible by ",
                    k_vector_size);

        // V: [NumBlocks, NumHeads, PageSize/kVector_size, HeadDim, kVector_size]
        head_size_v = v.size(3);
        num_blocks  = k.size(0);
    }
    else if(k.dim() == 4)
    {
        kv_memory_layout = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT;
        TORCH_CHECK(v.dim() == 4,
                    "V tensor must be 4D [NumBlocks, PageSize, NumHeads, HeadDim]");

        // K/V: [NumBlocks, PageSize, NumHeads, HeadDim]
        num_heads_k     = k.size(2);
        page_block_size = k.size(1);
        head_size_v     = v.size(3);
        num_blocks      = k.size(0);
    }
    else if(k.dim() == 3)
    {
        kv_memory_layout = ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT;
        TORCH_CHECK(v.dim() == 3, "V tensor must be 3D [NumBlocks, NumHeads, HeadDim]");

        // K/V: [NumBlocks, NumHeads, HeadDim] (PageSize=1)
        num_heads_k     = k.size(1);
        page_block_size = 1;
        head_size_v     = v.size(2);
        num_blocks      = k.size(0);
    }
    else
    {
        TORCH_CHECK(false,
                    "K tensor must be 5D (vectorized), 4D (linear), or 3D (linear, page_size=1) "
                    "for batch prefill");
    }


    if(max_seqlen_q == 1 && !alibi_slopes_.has_value())
    {
        is_causal = false;
    } // causal=true is the same as causal=false in this case

    TORCH_CHECK(!(bias_.has_value() && alibi_slopes_.has_value()),
                "cannot apply bias and alibi at the same time");
    bias_enum bias_type = bias_.has_value()           ? bias_enum::elementwise_bias
                          : alibi_slopes_.has_value() ? bias_type = bias_enum::alibi
                                                      : bias_enum::no_bias;

    // TODO
    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in
    // this case H/t Daniel Haziza

    const int total_q = q.size(0);

    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_q <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_v <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_q % k_vector_size == 0,
                "query, key, value, and out_ must have a head_size that is a multiple of ",
                k_vector_size);
    TORCH_CHECK(head_size_v % k_vector_size == 0,
                "query, key, value, and out_ must have a head_size that is a multiple of ",
                k_vector_size);
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    if(window_size_left >= max_seqlen_k)
    {
        window_size_left = -1;
    }
    if(window_size_right >= max_seqlen_k)
    {
        window_size_right = -1;
    }

    mask_info mask;

    if(is_causal)
    {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right         = 0;
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0" + "," + std::to_string(sink_size);
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // casual
    }
    else if(window_size_left == -1 && window_size_right == -1)
    {
        mask = mask_info::decode("0", max_seqlen_q, max_seqlen_k); // no mask; sink N/A for full attention
    }
    else
    {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify =
            "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right) + "," + std::to_string(sink_size);
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // local
    }

    CHECK_SHAPE(q, total_q, num_heads, head_size_q);
    
    if(kv_memory_layout == ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT)
    {
        // K: [NumBlocks, NumHeads, HeadDim/k_vector_size, PageSize, k_vector_size]
        CHECK_SHAPE(k,
                    num_blocks,
                    num_heads_k,
                    head_size_q / k_vector_size,
                    page_block_size,
                    k_vector_size);
        // V: [NumBlocks, NumHeads, PageSize/k_vector_size, HeadDim, k_vector_size]
        CHECK_SHAPE(v,
                    num_blocks,
                    num_heads_k,
                    page_block_size / k_vector_size,
                    head_size_v,
                    k_vector_size);
    }
    else
    {
        if(k.dim() == 3)
        {
            // K/V: [NumBlocks, NumHeads, HeadDim] (PageSize=1)
            CHECK_SHAPE(k, num_blocks, num_heads_k, head_size_q);
            CHECK_SHAPE(v, num_blocks, num_heads_k, head_size_v);
        }
        else
        {
            // K/V: [NumBlocks, PageSize, NumHeads, HeadDim]
            CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size_q);
            CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size_v);
        }
    }

    if(page_block_size > 1 && !block_table_.has_value())
    {
        TORCH_CHECK(kv_last_page_lens_.has_value(),
                    "if page_block_size > 1, must pass kv_last_page_lens to kernel");
    }

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(kv_indptr, batch_size + 1);
    auto opts = q.options();

    // This private module only contains the four validated BF16/D128/page16 arms.
    TORCH_CHECK(q_dtype == at::ScalarType::BFloat16 &&
                    head_size_q == 128 && head_size_v == 128 &&
                    num_heads == 16 && num_heads_k == 8 && page_block_size == 16 &&
                    k.dim() == 5 && v.dim() == 5 &&
                    block_table_.has_value() && seqlen_k_.has_value() &&
                    max_seqlen_q > 0 && max_seqlen_q <= 128 &&
                    p_dropout == 0.0f && logits_soft_cap == 0.0f &&
                    !is_causal && window_size_left == -1 && window_size_right == -1 &&
                    !bias_.has_value() && !alibi_slopes_.has_value() &&
                    qscale_type == quant_scale_enum::no_scale &&
                    sink_size == 0 && !sink_ptr.has_value() &&
                    return_softmax_lse && !return_dropout_randval,
                "Private FP32-O module requires BF16 Q16/KV8 D128 page16 vectorized "
                "VLLM, LSE, and no mask/bias/dropout/quant-scale/sink");
    const auto out_type = at::ScalarType::Float;
    at::Tensor out;
    if(out_.has_value())
    {
        out = out_.value();
        TORCH_CHECK(out.dtype() == out_type,
                    "Private BF16-input prefix requires FP32 output");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size_v);
    }
    else
    {
        out = torch::empty({total_q, num_heads, head_size_v}, opts.dtype(out_type));
    }

    // Otherwise the kernel will be launched from cuda:0 device
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q.device()};
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    bool has_lse     = return_softmax_lse;
    bool has_dropout = p_dropout > 0.0f;

    at::Tensor softmax_lse;
    if(return_softmax_lse)
    {
        softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat32));
    }
    else
    {
        softmax_lse = torch::empty({0}, opts.dtype(torch::kFloat32));
    }

    at::Tensor p;
    if(return_dropout_randval)
    {
        TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
        p = torch::empty({num_heads, total_q, max_seqlen_k}, opts.dtype(torch::kUInt8));
    }
    else
    {
        p = torch::empty({0}, opts);
    }

    if(zero_tensors)
    {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if(return_dropout_randval)
        {
            p.zero_();
        }
    }

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    auto rng_state         = torch::empty({2}, opts.dtype(torch::kInt64));
    auto rng_state_ptr     = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if(p_dropout > 0.0)
    {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(
            aiter::ParsePhiloxCudaState, dim3(1), dim3(64), 0, stream, philox_args, rng_state_ptr);
    }

    if(max_seqlen_k > 0)
    {
        ck_tile::stream_config stream_config{stream};

        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);

        auto args = get_ck_fmha_batch_prefill_args(has_lse,
                                                   return_dropout_randval,
                                                   mask,
                                                   batch_size,
                                                   max_seqlen_q,
                                                   num_heads,
                                                   num_heads_k,
                                                   head_size_q,
                                                   head_size_v,
                                                   num_blocks,
                                                   page_block_size,
                                                   kv_memory_layout,
                                                   q,
                                                   k,
                                                   v,
                                                   cu_seqlens_q,
                                                   kv_indptr,
                                                   kv_page_indices,
                                                   sink_ptr,
                                                   bias_,
                                                   alibi_slopes_,
                                                   q_descale,
                                                   k_descale,
                                                   v_descale,
                                                   kv_block_descale,
                                                   out,
                                                   softmax_lse,
                                                   p,
                                                   softmax_scale,
                                                   logits_soft_cap,
                                                   p_dropout,
                                                   drop_seed_offset,
                                                   kv_last_page_lens_);

        if(block_table_.has_value())
        {
            auto block_table = block_table_.value();
            CHECK_DEVICE(block_table);
            TORCH_CHECK(block_table.scalar_type() == at::kInt,
                        "block_table must be int32");
            TORCH_CHECK(block_table.dim() == 2, "block_table must be 2d");
            TORCH_CHECK(block_table.size(0) == batch_size,
                        "block_table first dim must match batch_size");
            TORCH_CHECK(block_table.stride(-1) == 1,
                        "block_table must have contiguous last dimension");
            TORCH_CHECK(seqlen_k_.has_value(),
                        "block_table requires seqlen_k for per-batch lengths");

            auto seqlen_k = seqlen_k_.value();
            CHECK_DEVICE(seqlen_k);
            TORCH_CHECK(seqlen_k.scalar_type() == at::kInt,
                        "seqlen_k must be int32");
            TORCH_CHECK(seqlen_k.dim() == 1, "seqlen_k must be 1d");
            TORCH_CHECK(seqlen_k.size(0) == batch_size,
                        "seqlen_k must have shape [batch_size]");

            args.kv_page_indices = block_table.data_ptr();
            args.batch_stride_block_table = block_table.stride(0);
            args.seqlen_k_ptr = seqlen_k.data_ptr();
            args.kv_lookup_table =
                ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D;
        }

        float t = aiter::mha_batch_prefill(args,
                                           stream_config,
                                           dtype_str,
                                           true, // is_group_mode
                                           mask.type,
                                           bias_type,
                                           has_lse,
                                           qscale_type,
                                           false);
        TORCH_CHECK(t >= 0,
                    "invalid argument for batch_prefill: no matching kernel found. "
                    "page_size=", args.page_block_size,
                    ", num_pages=", args.num_total_pages,
                    ", dtype=", dtype_str,
                    ". If KV cache exceeds 2GB (INT32_MAX byte offset) with page_size < kN0, "
                    "CDNA3+ GPU (MI300/MI350) is required.");
    }
    else
    {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return {out, softmax_lse, p, rng_state};
}

} // namespace torch_itfs
} // namespace aiter
