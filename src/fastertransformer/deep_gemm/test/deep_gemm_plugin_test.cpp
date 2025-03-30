#include <gtest/gtest.h>
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "maga_transformer/cpp/utils/SignalUtils.h"
#include <nvtx3/nvToolsExt.h>
#include <cuda_fp8.h>

#include "src/fastertransformer/deep_gemm/DeepGemmPlugin.h"

using namespace fastertransformer;
using namespace rtp_llm;

class DeepGemmPluginTest : public DeviceTestBase {
public:
    void RunDeepGeemPluginTest() {
        int m = 128, n = 2048, k = 7168;

        auto input =  torch::randn({(int)m, (int)k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto input_scale = torch::rand({m, int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto weight = torch::randn({n, k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto weight_scale = torch::randn({int(n / 128), int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto input_scale_t = input_scale.repeat({1, 128}).reshape({1, m, 128, (int)(k / 128)}).permute({1, 0, 3, 2}).reshape({m, k});
        auto weight_scale_t = weight_scale.repeat({128, 128}).reshape({(int)128, (int)(n / 128), 128, (int)(k / 128)}).permute({1, 0, 3, 2}).reshape({(int)(n), (int)(k)});
        auto ref_output = torch::matmul(input.to(torch::kFloat32) * input_scale_t, torch::transpose((weight.to(torch::kFloat32) * weight_scale_t), 0, 1));
        printBufferData(*fastertransformer::torchTensor2Buffer(ref_output), "ref", device_, true);

        BufferPtr input_k, input_s;
        input_k = torchTensor2Buffer(input);
        input_s = torchTensor2Buffer(input_scale);
        BufferPtr lhs = std::make_shared<QBuffer>(std::move(input_k), std::move(input_s), std::move(BufferPtr(new Buffer(input_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));

        BufferPtr weight_k, weight_s;
        weight_k = torchTensor2Buffer(weight);
        weight_s = torchTensor2Buffer(weight_scale);
        BufferPtr rhs = std::make_shared<QBuffer>(std::move(weight_k), std::move(weight_s), std::move(BufferPtr(new Buffer(weight_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));
        BufferPtr output = device_->allocateBuffer({DataType::TYPE_BF16, {(unsigned long)m, (unsigned long)n}, AllocationType::DEVICE});

        DeepGemmPlugin::gemmFp8(*lhs, *rhs, *output, 0);
        auto gemm_output = torch::from_blob(output->data(), {(int64_t)m, (int64_t)n}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        printBufferData(*output, "output", device_, true);
        auto sum = torch::sum(ref_output * ref_output + (gemm_output * gemm_output)).to(torch::kFloat32);
        EXPECT_NEAR(1, (2 * torch::sum(ref_output * gemm_output) / sum).item<double>(), 0.001);
    }
    void RunDeepGeemPluginGroupedContiguousTest() {
        int m = 128, n = 4096, k = 7168, num_groups = 256;

        auto input =  torch::ones({(int)num_groups, (int)m, (int)k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto input_scale = torch::randn({num_groups, m, int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto weight = torch::randn({num_groups, n, k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto weight_scale = torch::randn({num_groups, int(n / 128), int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto input_scale_t = input_scale.repeat({1, 1, 128}).reshape({num_groups, m, 128, (int)(k / 128)}).permute({0, 1, 3, 2}).reshape({num_groups, m, k});
        auto weight_scale_t = weight_scale.repeat({1, 128, 128}).reshape({num_groups, (int)128, (int)(n / 128), 128, (int)(k / 128)}).permute({0, 2, 1, 4, 3}).reshape({(int)num_groups, (int)(n), (int)(k)});
        auto weight_scale_tt = (weight.to(torch::kFloat32) * weight_scale_t).index({torch::indexing::Slice(0,1), torch::indexing::Slice(), torch::indexing::Slice()}).transpose(1,2).reshape({k, n});
        for (auto& x: weight_scale_tt.sizes()) std::cout << x << " ";
        std::cout << std::endl;
        auto ref_output = torch::matmul(input.to(torch::kFloat32) * input_scale_t, weight_scale_tt).reshape({num_groups * m, n});
        // auto ref_output = torch::einsum("gmk,gnk->gmn", {input.to(torch::kFloat32) * input_scale_t, (weight.to(torch::kFloat32) * weight_scale_t)}).reshape({num_groups * m, n});
        printBufferData(*fastertransformer::torchTensor2Buffer(ref_output), "ref", device_, true);

        BufferPtr input_k, input_s;
        input_k = torchTensor2Buffer(input.reshape({num_groups * m, k}));
        input_s = torchTensor2Buffer(input_scale.reshape({num_groups * m, k / 128}));
        BufferPtr lhs = std::make_shared<QBuffer>(std::move(input_k), std::move(input_s), std::move(BufferPtr(new Buffer(input_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));

        BufferPtr weight_k, weight_s;
        weight_k = torchTensor2Buffer(weight);
        weight_s = torchTensor2Buffer(weight_scale);

        //auto m_indices = torch::arange(0, num_groups, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        //m_indices = m_indices.unsqueeze(-1).expand({num_groups, m}).contiguous().reshape({num_groups * m});
        auto m_indices = torch::zeros({num_groups * m}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

        BufferPtr rhs = std::make_shared<QBuffer>(std::move(weight_k), std::move(weight_s), std::move(BufferPtr(new Buffer(weight_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));
        BufferPtr output = device_->allocateBuffer({DataType::TYPE_BF16, {(unsigned long)m * num_groups, (unsigned long)n}, AllocationType::DEVICE});

        DeepGemmPlugin::groupedGemmFp8Contiguous(*lhs, *rhs, *output, *(torchTensor2Buffer(m_indices)), 0);
        auto gemm_output = torch::from_blob(output->data(), {(int64_t)m * num_groups, (int64_t)n}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
        auto sum = torch::sum(ref_output * ref_output + (gemm_output * gemm_output)).to(torch::kFloat32);
        printBufferData(*output, "output", device_, true);
        std::cout << 2 * torch::sum(ref_output * gemm_output) / sum << std::endl;
        EXPECT_NEAR(1, (2 * torch::sum(ref_output * gemm_output) / sum).item<double>(), 0.001);
    }
};

TEST_F(DeepGemmPluginTest, Test1) {
    // RunDeepGeemPluginTest();
    RunDeepGeemPluginGroupedContiguousTest();
    RunDeepGeemPluginGroupedContiguousTest();
}
