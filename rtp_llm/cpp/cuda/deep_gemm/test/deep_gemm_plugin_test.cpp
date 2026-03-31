#include <gtest/gtest.h>
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include <nvtx3/nvToolsExt.h>
#include <cuda_fp8.h>

#include "rtp_llm/cpp/cuda/deep_gemm/DeepGemmPlugin.h"
#include "rtp_llm/cpp/cuda/deep_gemm/JIT.h"

using namespace rtp_llm;

class DeepGemmPluginTest: public DeviceTestBase {
public:
    void calcDiff(const torch::Tensor& x, const torch::Tensor& y) {
        auto sum = torch::sum(x * x + (y * y)).to(torch::kFloat32);
        EXPECT_NEAR(1, (2 * torch::sum(x * y) / sum).item<double>(), 0.1);
    }

    void RunDeepGemmPluginTest(int m, int n, int k) {
        auto input        = torch::randn({(int)m, (int)k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto input_scale  = torch::rand({m, int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto weight       = torch::randn({n, k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto weight_scale = torch::randn({int(n / 128), int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto input_scale_t =
            input_scale.repeat({1, 128}).reshape({1, m, 128, (int)(k / 128)}).permute({1, 0, 3, 2}).reshape({m, k});
        auto weight_scale_t = weight_scale.repeat({128, 128})
                                  .reshape({(int)128, (int)(n / 128), 128, (int)(k / 128)})
                                  .permute({1, 0, 3, 2})
                                  .reshape({(int)(n), (int)(k)});
        auto ref_output = torch::matmul(input.to(torch::kFloat32) * input_scale_t,
                                        torch::transpose((weight.to(torch::kFloat32) * weight_scale_t), 0, 1));

        auto lhs_kernel = input;
        auto lhs_scales = input_scale.transpose(0, 1).contiguous();
        auto rhs_kernel = weight;
        auto rhs_scales = weight_scale;

        auto output =
            torch::empty({(int64_t)m, (int64_t)n}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

        DeepGemmPlugin::gemmFp8(lhs_kernel, lhs_scales, rhs_kernel, rhs_scales, output, -1, 0);
        calcDiff(ref_output, output);
    }
    void RunDeepGemmPluginGroupedContiguousTest(int m, int n, int k, int num_groups) {
        auto input =
            torch::ones({(int)num_groups, (int)m, (int)k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto input_scale = torch::randn({num_groups, m, int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto weight      = torch::randn({num_groups, n, k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto weight_scale =
            torch::randn({num_groups, int(n / 128), int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto input_scale_t = input_scale.repeat({1, 1, 128})
                                 .reshape({num_groups, m, 128, (int)(k / 128)})
                                 .permute({0, 1, 3, 2})
                                 .reshape({num_groups, m, k});
        auto weight_scale_t = weight_scale.repeat({1, 128, 128})
                                  .reshape({num_groups, (int)128, (int)(n / 128), 128, (int)(k / 128)})
                                  .permute({0, 2, 1, 4, 3})
                                  .reshape({(int)num_groups, (int)(n), (int)(k)});
        auto weight_scale_tt =
            (weight.to(torch::kFloat32) * weight_scale_t)
                .index({torch::indexing::Slice(0, 1), torch::indexing::Slice(), torch::indexing::Slice()})
                .transpose(1, 2)
                .reshape({k, n});
        auto ref_output =
            torch::einsum("gmk,gnk->gmn",
                          {input.to(torch::kFloat32) * input_scale_t, (weight.to(torch::kFloat32) * weight_scale_t)})
                .reshape({num_groups * m, n});

        auto lhs_kernel = input.reshape({num_groups * m, k});
        auto lhs_scales = input_scale.reshape({num_groups * m, k / 128});
        auto rhs_kernel = weight;
        auto rhs_scales = weight_scale;

        auto m_indices = torch::arange(0, num_groups, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        m_indices      = m_indices.unsqueeze(-1).expand({num_groups, m}).contiguous().reshape({num_groups * m});

        auto output = torch::empty({(int64_t)m * num_groups, (int64_t)n},
                                   torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

        DeepGemmPlugin::groupedGemmFp8Contiguous(
            lhs_kernel, lhs_scales, rhs_kernel, rhs_scales, output, m_indices, -1, 0, 0);
        calcDiff(ref_output, output);
    }

    void
    RunDeepGeemPluginGroupedMaskedTest(int m, int n, int k, int num_groups, int expected_m = 0, bool run_v2 = false) {
        auto input = torch::randn({(int)1, (int)m, (int)k}, torch::device(torch::kCUDA))
                         .to(torch::kFloat8_e4m3fn)
                         .repeat({num_groups, 1, 1})
                         .contiguous();
        auto input_scale = torch::randn({1, m, int(k / 128)}, torch::device(torch::kCUDA))
                               .to(torch::kFloat32)
                               .repeat({num_groups, 1, 1})
                               .contiguous();

        auto weight = torch::randn({num_groups, n, k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);

        auto weight_scale =
            torch::randn({num_groups, int(n / 128), int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto input_scale_t = input_scale.repeat({1, 1, 128})
                                 .reshape({num_groups, m, 128, (int)(k / 128)})
                                 .permute({0, 1, 3, 2})
                                 .reshape({num_groups, m, k});
        auto weight_scale_t = weight_scale.repeat({1, 128, 128})
                                  .reshape({num_groups, (int)128, (int)(n / 128), 128, (int)(k / 128)})
                                  .permute({0, 2, 1, 4, 3})
                                  .reshape({(int)num_groups, (int)(n), (int)(k)});
        auto weight_scale_tt = (weight.to(torch::kFloat32) * weight_scale_t).transpose(1, 2);

        auto ref_output =
            torch::matmul(input.to(torch::kFloat32) * input_scale_t, weight_scale_tt).reshape({num_groups, m, n});

        auto lhs_kernel = input.reshape({num_groups, m, k});
        auto lhs_scales = input_scale.reshape({num_groups, m, k / 128}).transpose(1, 2).contiguous();
        auto rhs_kernel = weight;
        auto rhs_scales = weight_scale;

        auto masked_m = torch::ones({num_groups}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)) * m;

        auto output = torch::zeros({(int64_t)num_groups, (int64_t)m, (int64_t)n},
                                   torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        if (!expected_m) {
            expected_m = m;
        }
        if (run_v2) {
            DeepGemmPlugin::groupedGemmFp8Masked_V2(
                lhs_kernel, lhs_scales, rhs_kernel, rhs_scales, output, masked_m, expected_m, -1, 0);
        } else {
            DeepGemmPlugin::groupedGemmFp8Masked(
                lhs_kernel, lhs_scales, rhs_kernel, rhs_scales, output, masked_m, expected_m, -1, 0);
        }
        for (int i = 0; i < num_groups; ++i) {
            calcDiff(
                ref_output.index(
                    {torch::indexing::Slice(i, i + 1), torch::indexing::Slice(0, i + 1), torch::indexing::Slice()}),
                output.index(
                    {torch::indexing::Slice(i, i + 1), torch::indexing::Slice(0, i + 1), torch::indexing::Slice()}));
        }
    }
};

TEST_F(DeepGemmPluginTest, NormalTest) {
    RunDeepGemmPluginTest(128, 4096, 7168);
}

TEST_F(DeepGemmPluginTest, JITTest) {
    RunDeepGemmPluginTest(128, 512, 1024);
    RunDeepGemmPluginTest(128, 7168, 4096);
}

TEST_F(DeepGemmPluginTest, GroupedTest) {
    RunDeepGemmPluginGroupedContiguousTest(128, 4096, 7168, 2);
    RunDeepGeemPluginGroupedMaskedTest(16, 3072, 4096, 16);
    RunDeepGeemPluginGroupedMaskedTest(128, 3072, 4096, 16);
    RunDeepGeemPluginGroupedMaskedTest(120, 6144, 8192, 5);
}
