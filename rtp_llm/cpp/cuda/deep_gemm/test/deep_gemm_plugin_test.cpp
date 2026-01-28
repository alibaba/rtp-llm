#include <gtest/gtest.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
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
        printBufferData(*rtp_llm::torchTensor2Buffer(ref_output), "ref");

        BufferPtr input_k, input_s;
        input_k       = torchTensor2Buffer(input);
        input_s       = torchTensor2Buffer(input_scale.transpose(0, 1).contiguous());
        BufferPtr lhs = std::make_shared<QBuffer>(
            std::move(input_k),
            std::move(input_s),
            std::move(BufferPtr(new Buffer(input_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));

        BufferPtr weight_k, weight_s;
        weight_k      = torchTensor2Buffer(weight);
        weight_s      = torchTensor2Buffer(weight_scale);
        BufferPtr rhs = std::make_shared<QBuffer>(
            std::move(weight_k),
            std::move(weight_s),
            std::move(BufferPtr(new Buffer(weight_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));
        BufferPtr output = device_->allocateBuffer(
            {DataType::TYPE_BF16, {(unsigned long)m, (unsigned long)n}, AllocationType::DEVICE});

        DeepGemmPlugin::gemmFp8(*lhs, *rhs, *output, -1, 0);
        auto gemm_output = torch::from_blob(output->data(),
                                            {(int64_t)m, (int64_t)n},
                                            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        printBufferData(*output, "output");
        calcDiff(ref_output, gemm_output);
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
        printBufferData(*rtp_llm::torchTensor2Buffer(ref_output), "ref");

        BufferPtr input_k, input_s;
        input_k       = torchTensor2Buffer(input.reshape({num_groups * m, k}));
        input_s       = torchTensor2Buffer(input_scale.reshape({num_groups * m, k / 128}));
        BufferPtr lhs = std::make_shared<QBuffer>(
            std::move(input_k),
            std::move(input_s),
            std::move(BufferPtr(new Buffer(input_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));

        BufferPtr weight_k, weight_s;
        weight_k = torchTensor2Buffer(weight);
        weight_s = torchTensor2Buffer(weight_scale);

        auto m_indices = torch::arange(0, num_groups, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        m_indices      = m_indices.unsqueeze(-1).expand({num_groups, m}).contiguous().reshape({num_groups * m});

        BufferPtr rhs = std::make_shared<QBuffer>(
            std::move(weight_k),
            std::move(weight_s),
            std::move(BufferPtr(new Buffer(weight_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));
        BufferPtr output = device_->allocateBuffer(
            {DataType::TYPE_BF16, {(unsigned long)m * num_groups, (unsigned long)n}, AllocationType::DEVICE});

        DeepGemmPlugin::groupedGemmFp8Contiguous(*lhs, *rhs, *output, *(torchTensor2Buffer(m_indices)), -1, 0, 0);
        auto gemm_output = torch::from_blob(output->data(),
                                            {(int64_t)m * num_groups, (int64_t)n},
                                            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto sum         = torch::sum(ref_output * ref_output + (gemm_output * gemm_output)).to(torch::kFloat32);
        printBufferData(*output, "output");
        calcDiff(ref_output, gemm_output);
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

        BufferPtr input_k, input_s;
        input_k = torchTensor2Buffer(input.reshape({num_groups, m, k}));
        input_s = torchTensor2Buffer(input_scale.reshape({num_groups, m, k / 128}).transpose(1, 2).contiguous());

        BufferPtr lhs = std::make_shared<QBuffer>(
            std::move(input_k),
            std::move(input_s),
            std::move(BufferPtr(new Buffer(input_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));

        BufferPtr weight_k, weight_s;
        weight_k = torchTensor2Buffer(weight);
        weight_s = torchTensor2Buffer(weight_scale);

        auto masked_m = torch::ones({num_groups}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)) * m;

        BufferPtr rhs = std::make_shared<QBuffer>(
            std::move(weight_k),
            std::move(weight_s),
            std::move(BufferPtr(new Buffer(weight_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));
        BufferPtr output = device_->allocateBuffer(
            {DataType::TYPE_BF16, {(unsigned long)num_groups, (size_t)m, (unsigned long)n}, AllocationType::DEVICE});
        device_->bufMemset(*output, 0);
        auto masked_m_b = torchTensor2Buffer(masked_m);
        if (!expected_m) {
            expected_m = m;
        }
        if (run_v2) {
            DeepGemmPlugin::groupedGemmFp8Masked_V2(*lhs, *rhs, *output, *masked_m_b, expected_m, -1, 0);
        } else {
            DeepGemmPlugin::groupedGemmFp8Masked(*lhs, *rhs, *output, *masked_m_b, expected_m, -1, 0);
        }
        auto gemm_output = torch::from_blob(output->data(),
                                            {(int64_t)num_groups, (int64_t)m, (int64_t)n},
                                            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        for (int i = 0; i < num_groups; ++i) {
            auto slice_a = torchTensor2Buffer(ref_output.index(
                {torch::indexing::Slice(i, i + 1), torch::indexing::Slice(0, i + 2), torch::indexing::Slice()}));
            auto slice_b = torchTensor2Buffer(gemm_output.index(
                {torch::indexing::Slice(i, i + 1), torch::indexing::Slice(0, i + 2), torch::indexing::Slice()}));
            calcDiff(
                ref_output.index(
                    {torch::indexing::Slice(i, i + 1), torch::indexing::Slice(0, i + 1), torch::indexing::Slice()}),
                gemm_output.index(
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
