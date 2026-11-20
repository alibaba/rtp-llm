#pragma once

#include <limits>
#include <torch/python.h>

#include "sm120_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm120.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"
#include "sm120_padding.hpp"

namespace deep_gemm {

static torch::Tensor sm120_grouped_tma_output(const torch::Tensor& d, int block_m, bool masked) {
    const int m = static_cast<int>(d.size(-2));
    const int n = static_cast<int>(d.size(-1));
    const bool direct = d.scalar_type() == torch::kBFloat16 and d.stride(-1) == 1
        and reinterpret_cast<std::uintptr_t>(d.data_ptr()) % 16 == 0
        and d.stride(-2) >= n and (d.stride(-2) * d.element_size()) % 16 == 0
        and d.stride(-2) <= std::numeric_limits<int>::max() / 2 and n % 8 == 0
        and (not masked or (m % block_m == 0 and d.stride(0) == m * d.stride(1)));
    if (direct)
        return d;
    if (masked)
        return torch::empty({d.size(0), align(m, block_m), align(n, 64)}, d.options());
    return torch::empty({align(m, block_m), align(n, 64)}, d.options());
}

class SM120FP8FP4Gemm1D1DRuntime final: public SM120LaunchRuntime<SM120FP8FP4Gemm1D1DRuntime> {
public:
    struct Args {
        SM120GemmDesc gemm_desc;
        SM120GemmConfig gemm_config;
        SM120LaunchArgs launch_args;
        const std::optional<std::string> epilogue_type;
        EpilogueArgs epilogue_args{};
        int stride_c_m = 0;

        int gran_k_a, gran_k_b;
        bool is_fp4;
        bool b_is_fp4;
        bool a_is_fp4;
        bool k_grouped_constant_stride;
        int k_alignment = 128;
        int stride_cd_m;
        int stride_cd_n;
        int stride_cd_batch;
        int shape_cd_m = 0;

        void* gmem_d;
        void* gmem_c;
        void* gmem_a_ptr;
        void* gmem_b_ptr;
        void* gmem_workspace;
        void* grouped_layout;
        void* tensor_map_buffer;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_fp4_gemm_1d1d_impl<
        {}, {}, {},
        {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gran_k_a, args.gran_k_b,
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.launch_config.num_sms,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        to_string(args.gemm_desc.cd_dtype),
        get_default_epilogue_type(args.epilogue_type),
        args.is_fp4 ? "true" : "false",
        args.b_is_fp4 ? "true" : "false",
        args.a_is_fp4 ? "true" : "false",
        (args.gemm_desc.major_b == cute::UMMA::Major::K) ? "true" : "false",
        args.k_grouped_constant_stride ? "true" : "false",
        args.gemm_config.storage_config.store_block_m,
        args.gemm_config.split_k_factor,
        args.k_alignment);
    }

    template <typename Kernel>
    static void launch_impl(const Kernel& kernel, const Args& args) {
        sm120_launch_kernel(kernel, args.launch_args,
            args.gmem_d, args.gmem_c,
            args.gmem_a_ptr, args.gmem_b_ptr,
            args.grouped_layout,
            args.tensor_map_buffer,
            args.gmem_workspace,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.stride_cd_m, args.stride_cd_n, args.stride_cd_batch,
            args.shape_cd_m != 0 ? args.shape_cd_m : args.gemm_desc.m,
            args.epilogue_args, args.stride_c_m,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_cd);
    }
};

static void sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                         const torch::Tensor& b, const torch::Tensor& sfb,
                                                         const torch::Tensor& d,
                                                         const torch::Tensor& grouped_layout,
                                                         const int& num_groups, const int& m, const int& n, const int& k,
                                                         const int& gran_k_a, const int& gran_k_b,
                                                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                         const std::string& compiled_dims,
                                                         const bool& use_psum_layout,
                                                         const std::optional<int>& expected_m_for_psum_layout) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    const bool a_dt_fp4 = (a.scalar_type() == kPackedFP4);
    const bool b_dt_fp4 = (b.scalar_type() == kPackedFP4);
    const bool is_fp4 = a_dt_fp4 && b_dt_fp4;       // symmetric FP4xFP4 (k64)
    const bool b_is_fp4 = (!a_dt_fp4) && b_dt_fp4;  // FP8_A x FP4_B (k32 mixed)
    const bool a_is_fp4 = a_dt_fp4 && (!b_dt_fp4);  // FP4_A x FP8_B (k32 mixed, swapAB)

    const auto desc = SM120GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = deep_gemm::runtime->get_num_sms(),
        .tc_util = deep_gemm::runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .max_gran_k = std::max(gran_k_a, gran_k_b),
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_sm120_config<SM120ArchSpec>(desc);

    const bool fp4_unpacked = !is_fp4;
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode, 0, false, fp4_unpacked);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode, 0, false,
                                              b_is_fp4 ? true : fp4_unpacked);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k_a, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k_b, num_groups, 0);
    const int cd_store_m = config.storage_config.store_block_m > 0
        ? config.storage_config.store_block_m : config.layout.block_m;
    const auto output = SM120ArchSpec::uses_grouped_tma_output(desc)
        ? sm120_grouped_tma_output(d, config.layout.block_m, false) : d;
    const auto tensor_map_cd = make_tma_cd_desc(output, static_cast<int>(output.size(-2)), static_cast<int>(output.size(-1)),
                                                cd_store_m, config.layout.block_n,
                                                static_cast<int>(output.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    const SM120FP8FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = SM120LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gran_k_a = gran_k_a,
        .gran_k_b = gran_k_b,
        .is_fp4 = is_fp4,
        .b_is_fp4 = b_is_fp4,
        .a_is_fp4 = a_is_fp4,
        .k_grouped_constant_stride = false,
        .stride_cd_m = static_cast<int>(output.stride(-2)),
        .stride_cd_n = 0,
        .stride_cd_batch = 0,
        .shape_cd_m = static_cast<int>(output.size(-2)),
        .gmem_d = output.data_ptr(),
        .gmem_c = nullptr,
        .gmem_a_ptr = a.data_ptr(),
        .gmem_b_ptr = nullptr,
        .gmem_workspace = nullptr,
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM120FP8FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = jit->compile("sm120_m_grouped_fp8_fp4_gemm_contiguous_1d1d", code);
    SM120FP8FP4Gemm1D1DRuntime::launch(runtime, args);
    if (output.data_ptr() != d.data_ptr())
        sm120_copy_grouped_output(output, d, grouped_layout, m, n, num_groups, false, use_psum_layout,
                                  heuristics_runtime->get_mk_alignment_for_contiguous_layout());
}

} // namespace deep_gemm
