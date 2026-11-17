#pragma once

// Isolated SM120 GEMM host APIs. Shared gemm.hpp only forwards arch==12 here.

#include <c10/cuda/CUDAGuard.h>

#include "../../vendor/csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp"
#include "../../vendor/csrc/jit_kernels/impls/sm120_padding.hpp"
#include "layout.hpp"

namespace deep_gemm::sm120 {

static torch::Tensor to_k_major(const torch::Tensor& t, const cute::UMMA::Major& major,
                                const int& logical_mn) {
    if (major == cute::UMMA::Major::K)
        return t;
    if (t.scalar_type() != kPackedFP4)
        return t.contiguous();
    const int ndim = t.dim();
    DG_HOST_ASSERT(ndim == 2 or ndim == 3);
    const int mn_packed = t.size(-2);
    const int k = t.size(-1);
    DG_HOST_ASSERT(mn_packed * 2 == logical_mn and k % 2 == 0);
    auto lo = t.bitwise_and(0x0F);
    auto hi = t.to(torch::kByte).bitwise_right_shift(4).to(torch::kInt8).bitwise_and(0x0F);
    auto shape_full = t.sizes().vec();
    shape_full[ndim - 2] = logical_mn;
    auto codes = torch::empty(shape_full, t.options());
    using S = torch::indexing::Slice;
    codes.index_put_({torch::indexing::Ellipsis, S(0, torch::indexing::None, 2), S()}, lo);
    codes.index_put_({torch::indexing::Ellipsis, S(1, torch::indexing::None, 2), S()}, hi);
    auto shape_view = shape_full;
    shape_view[ndim - 1] = k / 2;
    shape_view.push_back(2);
    auto codes2 = codes.view(shape_view);
    return codes2.select(-1, 0).bitwise_and(0x0F)
           .bitwise_or(codes2.select(-1, 1).bitwise_and(0x0F).to(torch::kByte)
                       .bitwise_left_shift(4).to(torch::kInt8)).contiguous();
}

static void m_grouped_fp8_fp4_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 std::optional<std::tuple<int, int, int>> recipe,
                                                 std::optional<std::tuple<int, int>> recipe_a,
                                                 std::optional<std::tuple<int, int>> recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const bool& ensure_zero_padding,
                                                 const std::optional<int>& expected_m_for_psum_layout,
                                                 const std::optional<pybind11::object>& epilogue_class) {
    const c10::cuda::CUDAGuard device_guard(a.first.device());
    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    for (const auto& t: {a.second, b.first, b.second, d, grouped_layout})
        DG_HOST_ASSERT(t.is_cuda() and t.device() == a.first.device());
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_fp4_requires_k_major(a.first, b.first))
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(not disable_ue8m0_cast or (a.second.scalar_type() == torch::kInt and b.second.scalar_type() == torch::kInt));
    const auto [m, k] = check_ab_fp8_fp4(a.first, major_a, 12);
    const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, major_b, 12);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    if (use_psum_layout) {
        DG_HOST_ASSERT(num_groups == static_cast<int>(grouped_layout.numel()));
    } else {
        DG_HOST_ASSERT(m == static_cast<int>(grouped_layout.numel()));
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }
    check_major_type_cd(d);
    if (m == 0)
        return;
    DG_HOST_ASSERT(not epilogue_class.has_value() and "Custom epilogues require SM100");
    const std::optional<torch::Tensor> psum_sfa_layout = use_psum_layout ? std::make_optional(grouped_layout) : std::nullopt;
    const auto [sfa, sfb, gran_k_a, gran_k_b] = transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, num_groups, disable_ue8m0_cast,
        psum_sfa_layout);
    const auto b_data = to_k_major(b.first, major_b, n);
    sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a.first, sfa, b_data, sfb, d, grouped_layout,
                                                 num_groups, m, n, k, gran_k_a, gran_k_b,
                                                 major_a, cute::UMMA::Major::K, compiled_dims,
                                                 use_psum_layout, expected_m_for_psum_layout);
    if (ensure_zero_padding)
        sm120_clear_padding(d, grouped_layout, m, n, num_groups, false, use_psum_layout,
                            heuristics_runtime->get_mk_alignment_for_contiguous_layout());
}

} // namespace deep_gemm::sm120
