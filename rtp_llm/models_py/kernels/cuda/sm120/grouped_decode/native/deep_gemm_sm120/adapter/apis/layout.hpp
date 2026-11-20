#pragma once

// Isolated SM120 SF transforms. Shared layout.hpp only forwards arch==12 here.

#include "../utils.hpp"
#include "../../../csrc/jit_kernels/impls/smxx_layout.hpp"

namespace deep_gemm::sm120 {

static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                       const int& mn, const int& k,
                                                       const std::variant<std::tuple<int, int, int>,
                                                                          std::tuple<int, int>>& recipe,
                                                       const std::optional<int>& num_groups,
                                                       const std::optional<bool>& is_sfa,
                                                       const bool& disable_ue8m0_cast,
                                                       const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    int gran_mn, gran_k;
    if (auto p = std::get_if<std::tuple<int, int, int>>(&recipe)) {
        DG_HOST_ASSERT(is_sfa.has_value());
        gran_mn = is_sfa.value() ? std::get<0>(*p) : std::get<1>(*p);
        gran_k = std::get<2>(*p);
    } else if (auto p = std::get_if<std::tuple<int, int>>(&recipe)) {
        DG_HOST_ASSERT(not is_sfa.has_value());
        std::tie(gran_mn, gran_k) = *p;
    } else {
        DG_HOST_UNREACHABLE("Invalid recipe");
    }

    DG_HOST_ASSERT(not psum_layout.has_value() or not num_groups.has_value());
    if (psum_layout.has_value()) {
        DG_HOST_ASSERT(psum_layout->is_cuda() and psum_layout->device() == sf.device());
        DG_HOST_ASSERT(psum_layout->dim() == 1 and psum_layout->is_contiguous() and psum_layout->scalar_type() == torch::kInt);
    }
    DG_HOST_ASSERT(gran_mn > 0 and (gran_k == 32 or gran_k == 128));
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);
    if (sf.scalar_type() == torch::kFloat) {
        if (disable_ue8m0_cast and gran_k == 128) {
            if (gran_mn == 1)
                return get_mn_major_tma_aligned_tensor(sf);
            if (gran_mn == 128)
                return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);
        }
        DG_HOST_ASSERT(not disable_ue8m0_cast and "SM120 MMA requires UE8M0 scales");
        const auto broadcasted = gran_mn == 1 ? sf :
            sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(gran_mn));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted, psum_layout);
    }
    DG_HOST_ASSERT(gran_mn == 1 and "SM120 prepacked UE8M0 scales must already be per-row");
    return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);
}

static std::tuple<torch::Tensor, torch::Tensor, int, int> transform_sf_pair_into_required_layout(
        const torch::Tensor& sfa, const torch::Tensor& sfb,
        const int& m, const int& n, const int& k,
        std::optional<std::tuple<int, int, int>>& recipe,
        const std::optional<std::tuple<int, int>>& recipe_a,
        const std::optional<std::tuple<int, int>>& recipe_b,
        const std::optional<int>& num_groups_a,
        const std::optional<int>& num_groups_b,
        const bool& disable_ue8m0_cast = false,
        const std::optional<torch::Tensor>& psum_layout = std::nullopt) {
    if (not recipe_a.has_value() and not recipe.has_value())
        recipe = get_default_recipe(sfa.scalar_type(), sfb.scalar_type());
    DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
    DG_HOST_ASSERT(recipe_a.has_value() != recipe.has_value());
    const auto transformed_sfa = recipe.has_value()
        ? transform_sf_into_required_layout(sfa, m, k, recipe.value(), num_groups_a, true, disable_ue8m0_cast, psum_layout)
        : transform_sf_into_required_layout(sfa, m, k, recipe_a.value(), num_groups_a, std::nullopt, disable_ue8m0_cast, psum_layout);
    const auto transformed_sfb = recipe.has_value()
        ? transform_sf_into_required_layout(sfb, n, k, recipe.value(), num_groups_b, false, disable_ue8m0_cast)
        : transform_sf_into_required_layout(sfb, n, k, recipe_b.value(), num_groups_b, std::nullopt, disable_ue8m0_cast);
    const int gran_k_a = recipe_a.has_value() ? std::get<1>(recipe_a.value()) : std::get<2>(recipe.value());
    const int gran_k_b = recipe_b.has_value() ? std::get<1>(recipe_b.value()) : std::get<2>(recipe.value());
    return std::make_tuple(transformed_sfa, transformed_sfb, gran_k_a, gran_k_b);
}

} // namespace deep_gemm::sm120
