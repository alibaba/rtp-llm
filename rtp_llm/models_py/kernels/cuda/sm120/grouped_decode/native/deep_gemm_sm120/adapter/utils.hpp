#pragma once

#include <algorithm>
#include "../../csrc/utils/layout.hpp"

namespace deep_gemm::sm120 {

template <bool kRequireContiguousBatch = true>
static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    if constexpr (kRequireContiguousBatch) {
        if (dim == 3 and t.size(-2) != 0 and t.size(-1) != 0)
            DG_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    }
    DG_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}

template <bool kRequireContiguousBatch = true>
static cute::UMMA::Major get_major_type_ab(const torch::Tensor& t) {
    major_check<kRequireContiguousBatch>(t);
    return t.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
}

template <bool kRequireContiguousBatch = true>
static void check_major_type_cd(const torch::Tensor& t) {
    major_check<kRequireContiguousBatch>(t);
    DG_HOST_ASSERT(t.stride(-1) == 1);
}

static std::tuple<int, int> check_ab_fp8_fp4(const torch::Tensor& ab, const cute::UMMA::Major& major, const int& = 12) {
    auto [mn, k] = get_shape<2>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4);
        major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(mn, k);
}

static std::tuple<int, int, int> check_grouped_ab_fp8_fp4(const torch::Tensor& ab, const cute::UMMA::Major& major, const int& = 12) {
    auto [num_groups, mn, k] = get_shape<3>(ab);
    if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
        DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4);
        major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);
    }
    return std::make_tuple(num_groups, mn, k);
}

static std::tuple<int, int, int>
get_default_recipe(const torch::ScalarType& sfa_dtype, const torch::ScalarType& sfb_dtype) {
    DG_HOST_ASSERT(sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt);
    (void)sfa_dtype;
    return sfb_dtype == torch::kFloat ?
        std::make_tuple(1, 128, 128) :
        std::make_tuple(1, 1, 128);
}

static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m) {
    if (not expected_m.has_value() or expected_m.value() > 64)
        return 128;
    return expected_m.value() <= 32 ? 32 : 64;
}

} // namespace deep_gemm::sm120
