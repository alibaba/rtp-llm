#include <gtest/gtest.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <nvtx3/nvToolsExt.h>
#include <cuda_fp8.h>

#include "rtp_llm/cpp/cuda/deep_gemm/DeepGemmPlugin.h"
#include "rtp_llm/cpp/cuda/deep_gemm/JIT.h"

using namespace rtp_llm;

class DeepGemmDeterminismTest: public DeviceTestBase {
public:
    void calcDiff(const torch::Tensor& x, const torch::Tensor& y, const std::string& name = "") {
        auto diff       = torch::abs(x - y);
        auto max_diff   = diff.max().item<float>();
        auto diff_count = (diff > 1e-6).sum().item<int64_t>();
        std::cout << name << " - Max diff: " << max_diff << ", Different elements: " << diff_count << std::endl;
    }

    void RunDeepGemmDeterminismTest(int m, int n, int k, int num_runs = 10) {
        std::cout << "Running DeepGemm determinism test with " << num_runs << " runs" << std::endl;
        std::cout << "GEMM dimensions: M=" << m << ", K=" << k << ", N=" << n << std::endl;

        torch::manual_seed(42);

        auto input        = torch::randn({m, k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto input_scale  = torch::rand({m, int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);
        auto weight       = torch::randn({n, k}, torch::device(torch::kCUDA)).to(torch::kFloat8_e4m3fn);
        auto weight_scale = torch::randn({int(n / 128), int(k / 128)}, torch::device(torch::kCUDA)).to(torch::kFloat32);

        std::cout << "Input shape: " << input.sizes() << ", dtype: " << input.dtype() << std::endl;
        std::cout << "Input_scale shape: " << input_scale.sizes() << ", dtype: " << input_scale.dtype() << std::endl;
        std::cout << "Weight shape: " << weight.sizes() << ", dtype: " << weight.dtype() << std::endl;
        std::cout << "Weight_scale shape: " << weight_scale.sizes() << ", dtype: " << weight_scale.dtype() << std::endl;

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

        std::vector<torch::Tensor> outputs;

        for (int i = 0; i < num_runs; ++i) {
            device_->bufMemset(*output, 0);
            DeepGemmPlugin::gemmFp8(*lhs, *rhs, *output, -1, 0);
            device_->syncAndCheck();

            auto gemm_output = torch::from_blob(output->data(),
                                                {(int64_t)m, (int64_t)n},
                                                torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA))
                                   .clone();
            outputs.push_back(gemm_output);

            std::cout << "Run " << i + 1 << " completed" << std::endl;
        }

        std::cout << "\nComparing outputs across " << num_runs << " runs..." << std::endl;
        torch::Tensor reference = outputs[0];
        bool          all_match = true;

        for (size_t i = 1; i < outputs.size(); ++i) {
            calcDiff(reference, outputs[i], "Run 0 vs Run " + std::to_string(i));

            auto diff       = torch::abs(reference - outputs[i]);
            auto max_diff   = diff.max().item<float>();
            auto diff_count = (diff > 1e-6).sum().item<int64_t>();

            if (max_diff > 1e-6 || diff_count > 0) {
                all_match = false;
                std::cout << "  FAIL: Outputs differ!" << std::endl;

                auto diff_mask    = diff > 1e-6;
                auto diff_indices = torch::nonzero(diff_mask);
                if (diff_indices.size(0) > 0) {
                    std::cout << "  First differing index: ";
                    for (int j = 0; j < std::min(5, (int)diff_indices.size(0)); ++j) {
                        auto idx = diff_indices[j];
                        std::cout << "[";
                        for (int d = 0; d < idx.size(0); ++d) {
                            std::cout << idx[d].item<int64_t>();
                            if (d < idx.size(0) - 1)
                                std::cout << ", ";
                        }
                        std::cout << "] ";
                    }
                    std::cout << std::endl;
                }
            } else {
                std::cout << "  PASS: Outputs match exactly" << std::endl;
            }
        }

        EXPECT_TRUE(all_match) << "DeepGemm outputs are NOT deterministic across " << num_runs << " runs";

        if (all_match) {
            std::cout << "\nSUCCESS: DeepGemm is deterministic across " << num_runs << " runs!" << std::endl;
        } else {
            std::cout << "\nFAILURE: DeepGemm is NOT deterministic!" << std::endl;
        }
    }
};

TEST_F(DeepGemmDeterminismTest, TestDeterminismWithRandomTensors) {
    RunDeepGemmDeterminismTest(128, 4096, 7168, 20);
}

TEST_F(DeepGemmDeterminismTest, TestDeterminismSmall) {
    RunDeepGemmDeterminismTest(64, 512, 1024, 20);
}