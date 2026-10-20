#include "dsv41_flashmla_pdl.h"

#if defined(RTP_DSV41_FLASHMLA_PDL_ENABLED) && !defined(USE_ROCM)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh"
#endif

namespace torch_ext {
bool dsv41_flashmla_pdl_available() {
#if defined(RTP_DSV41_FLASHMLA_PDL_ENABLED) && !defined(USE_ROCM)
    return true;
#else
    return false;
#endif
}

void dsv41_flashmla_decode_out(torch::Tensor q, torch::Tensor pool, torch::Tensor indices,
    torch::Tensor extra, torch::Tensor extra_indices, torch::Tensor positions,
    torch::Tensor cos_sin, torch::Tensor sink, torch::Tensor output, torch::Tensor scales,
    torch::Tensor lse, double sm_scale, bool wait_pdl) {
#if defined(RTP_DSV41_FLASHMLA_PDL_ENABLED) && !defined(USE_ROCM)
    TORCH_CHECK(q.is_cuda(), "FlashMLA requires CUDA");
    const c10::cuda::CUDAGuard guard(q.device());
    const auto* props = at::cuda::getDeviceProperties(q.get_device());
    TORCH_CHECK(props->major == 10 && props->minor == 3 && props->multiProcessorCount == 148,
                "PDL FlashMLA is qualified for runtime SM103 / 148 SM");
    for (auto t : {pool, indices, extra, extra_indices, positions, cos_sin, sink, output, scales, lse})
        TORCH_CHECK(t.is_cuda() && t.device() == q.device(), "FlashMLA tensors must share a device");
    const int64_t m = q.size(0);
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 512
                && q.scalar_type() == at::kBFloat16 && q.is_contiguous(), "Invalid permuted Q");
    for (auto pair : {std::make_pair(pool, 528), std::make_pair(extra, 288)}) {
        auto t = pair.first;
        TORCH_CHECK(t.scalar_type() == at::kByte && t.dim() == 3 && t.size(0) > 0 && t.size(1) > 0
                    && t.size(2) == pair.second && t.stride(2) == 1 && t.stride(1) == pair.second
                    && t.stride(0) >= t.size(1) * pair.second && t.stride(0) % pair.second == 0
                    && uintptr_t(t.data_ptr()) % 16 == 0, "Invalid paged cache");
    }
    for (auto t : {indices, extra_indices})
        TORCH_CHECK(t.scalar_type() == at::kInt && t.dim() == 2 && t.size(0) == m && t.size(1) > 0
                    && t.size(1) % 128 == 0 && t.is_contiguous(), "Invalid physical slot indices");
    TORCH_CHECK(positions.scalar_type() == at::kInt && positions.dim() == 1
                && positions.numel() == m && positions.is_contiguous(), "Invalid token positions");
    TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.dim() == 2 && cos_sin.size(0) > 0
                && cos_sin.size(1) == 64 && cos_sin.is_contiguous(), "Invalid RoPE table");
    TORCH_CHECK(sink.scalar_type() == at::kFloat && sink.numel() == 64 && sink.is_contiguous(), "Invalid sink");
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Float8_e4m3fn && output.is_contiguous()
                && output.sizes() == at::IntArrayRef({m, 8, 4096}), "Invalid FP8 output");
    TORCH_CHECK(scales.scalar_type() == at::kInt && scales.sizes() == at::IntArrayRef({m, 8, 32})
                && scales.stride(0) == 1 && scales.stride(2) >= m && scales.stride(2) % 4 == 0
                && scales.stride(1) == 32 * scales.stride(2), "Invalid packed output SF");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous()
                && lse.sizes() == at::IntArrayRef({m, 64}), "Invalid LSE");
    TORCH_CHECK(std::isfinite(sm_scale) && sm_scale > 0, "Invalid softmax scale");
    for (auto o : {output, scales, lse}) {
        for (auto i : {q, pool, indices, extra, extra_indices, positions, cos_sin, sink})
            TORCH_CHECK(!o.is_alias_of(i), "FlashMLA output aliases input");
    }
    TORCH_CHECK(!output.is_alias_of(scales) && !output.is_alias_of(lse) && !scales.is_alias_of(lse),
                "FlashMLA outputs must have separate storage");
    using namespace sm100::prefill::rtp_dsv41_pdl_flash;
    ParamT<SparseAttnFwdMode::Decode> p{};
    p.b = 1; p.s_q = m; p.h_q = 64; p.h_kv = 1; p.d_qk = p.d_v = 512;
    p.sm_scale = sm_scale; p.sm_scale_div_log2 = p.sm_scale * 1.4426950408889634f;
    p.model_type = ModelType::V41; p.extra_model_type = ModelType::V41_FP4;
    p.q = reinterpret_cast<bf16*>(q.data_ptr()); p.kv = reinterpret_cast<bf16*>(pool.data_ptr());
    p.num_blocks = pool.size(0); p.page_block_size = pool.size(1); p.topk = indices.size(1);
    p.indices = indices.data_ptr<int>(); p.attn_sink = sink.data_ptr<float>(); p.lse = lse.data_ptr<float>();
    p.extra_kv = reinterpret_cast<bf16*>(extra.data_ptr()); p.extra_indices = extra_indices.data_ptr<int>();
    p.extra_num_blocks = extra.size(0); p.extra_page_block_size = extra.size(1); p.extra_topk = extra_indices.size(1);
    p.stride_q_s_q = q.stride(0); p.stride_q_h_q = q.stride(1);
    p.stride_kv_block = pool.stride(0); p.stride_kv_row = pool.stride(1);
    p.stride_extra_kv_block = extra.stride(0); p.stride_extra_kv_row = extra.stride(1);
    p.stride_indices_s_q = indices.stride(0); p.stride_extra_indices_s_q = extra_indices.stride(0);
    p.stride_lse_s_q = lse.stride(0); p.num_sm = props->multiProcessorCount;
    p.stream = at::cuda::getCurrentCUDAStream(); p.use_pdl = wait_pdl;
    p.enable_q_norm = false; p.rms_norm_eps = 1e-6f;
    p.token_positions = reinterpret_cast<uint32_t*>(positions.data_ptr<int>());
    p.is_rope_neox_style = false; p.rope_dim = 64; p.cos_sin_cache = cos_sin.data_ptr<float>();
    p.n_wv_group = 8; p.wv_group_size = 8; p.num_per_channels = 32;
    p.use_tma_aligned_col_major_sf = p.round_sf = p.use_packed_ue8m0 = true;
    p.out_fp8 = reinterpret_cast<fp8_e4m3*>(output.data_ptr()); p.out_sf = reinterpret_cast<uint32_t*>(scales.data_ptr<int>());
    p.stride_out_sf_wv_group = scales.stride(1); p.stride_out_sf_head_dim = scales.stride(2);
    run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode,
        ModelType::V41, ModelType::V41_FP4, 64, false}>(p);
#else
    TORCH_CHECK(false, "PDL FlashMLA was not built for this platform");
#endif
}
} // namespace torch_ext
