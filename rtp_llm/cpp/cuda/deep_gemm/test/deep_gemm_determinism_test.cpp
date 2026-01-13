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
    torch::Tensor loadTorchTensor(const std::string& pt_path, torch::Dtype target_dtype) {
        auto tensor = loadTensorFromFile(pt_path);

        if (target_dtype == torch::kFloat8_e4m3fn) {
            if (tensor.dtype() == torch::kChar) {
                tensor = tensor.view(target_dtype);
            }
        }

        return tensor.to(torch::kCUDA);
    }

    void calcDiff(const torch::Tensor& x, const torch::Tensor& y, const std::string& name = "") {
        auto diff       = torch::abs(x - y);
        auto max_diff   = diff.max().item<float>();
        auto diff_count = (diff > 1e-6).sum().item<int64_t>();
        std::cout << name << " - Max diff: " << max_diff << ", Different elements: " << diff_count << std::endl;
    }

    void RunDeepGemmDeterminismTestWithDumpedTensors(const std::string& pt_dir, int num_runs = 10) {
        std::cout << "Running DeepGemm determinism test with " << num_runs << " runs" << std::endl;
        std::cout << "Loading tensors from " << pt_dir << "..." << std::endl;

        auto a_kernel        = loadTorchTensor(pt_dir + "/A_kernel.pt", torch::kFloat8_e4m3fn);
        auto a_scales        = loadTorchTensor(pt_dir + "/A_scales.pt", torch::kFloat32);
        auto b_kernel        = loadTorchTensor(pt_dir + "/B_kernel.pt", torch::kFloat8_e4m3fn);
        auto b_scales        = loadTorchTensor(pt_dir + "/B_scales.pt", torch::kFloat32);
        auto expected_output = loadTorchTensor(pt_dir + "/gemm_output.pt", torch::kBFloat16);

        std::cout << "A kernel: " << a_kernel.sizes() << ", dtype: " << a_kernel.dtype() << std::endl;
        std::cout << "A scales: " << a_scales.sizes() << ", dtype: " << a_scales.dtype() << std::endl;
        std::cout << "B_kernel shape: " << b_kernel.sizes() << ", dtype: " << b_kernel.dtype() << std::endl;
        std::cout << "B_scales shape: " << b_scales.sizes() << ", dtype: " << b_scales.dtype() << std::endl;
        std::cout << "Expected output shape: " << expected_output.sizes() << ", dtype: " << expected_output.dtype()
                  << std::endl;

        auto m = a_kernel.size(0);
        auto k = a_kernel.size(1);
        auto n = b_kernel.size(1);

        std::cout << "GEMM dimensions: M=" << m << ", K=" << k << ", N=" << n << std::endl;

        auto expected_m = expected_output.size(0);
        auto expected_n = expected_output.size(1);
        std::cout << "Expected output dimensions: M=" << expected_m << ", N=" << expected_n << std::endl;

        if (expected_m != m || expected_n != n) {
            std::cout << "Slicing expected output from [" << expected_m << ", " << expected_n << "] to [" << m << ", "
                      << n << "]" << std::endl;
            expected_output = expected_output.slice(0, 0, m).slice(1, 0, n);
        }

        BufferPtr input_k, input_s;
        input_k       = torchTensor2Buffer(a_kernel);
        input_s       = torchTensor2Buffer(a_scales);
        BufferPtr lhs = std::make_shared<QBuffer>(
            std::move(input_k),
            std::move(input_s),
            std::move(BufferPtr(new Buffer(input_k->where(), DataType::TYPE_INVALID, {0}, nullptr))));

        BufferPtr weight_k, weight_s;
        weight_k      = torchTensor2Buffer(b_kernel);
        weight_s      = torchTensor2Buffer(b_scales);
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
            auto diff_count = (diff > 1e-16).sum().item<int64_t>();

            if (max_diff > 1e-16 || diff_count > 0) {
                all_match = false;
                std::cout << "  FAIL: Outputs differ!" << std::endl;

                auto diff_mask    = diff > 1e-16;
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

        std::cout << "\nComparing outputs against expected output..." << std::endl;
        bool          all_match_expected = true;
        torch::Tensor first_output       = outputs[0];

        calcDiff(first_output, expected_output, "Run 0 vs Expected output");

        auto diff       = torch::abs(first_output - expected_output);
        auto max_diff   = diff.max().item<float>();
        auto diff_count = (diff > 1e-6).sum().item<int64_t>();

        if (max_diff > 1e-6 || diff_count > 0) {
            all_match_expected = false;
            std::cout << "  FAIL: Output does NOT match expected!" << std::endl;

            auto diff_mask    = diff > 1e-6;
            auto diff_indices = torch::nonzero(diff_mask);
            if (diff_indices.size(0) > 0) {
                std::cout << "  First differing index: ";
                for (int j = 0; j < std::min(5120, (int)diff_indices.size(0)); ++j) {
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

            std::cout << "  Differing values:" << std::endl;
            for (int j = 0; j < std::min(5, (int)diff_indices.size(0)); ++j) {
                auto  idx     = diff_indices[j];
                int   row     = idx[0].item<int64_t>();
                int   col     = idx[1].item<int64_t>();
                float out_val = first_output[row][col].item<float>();
                float exp_val = expected_output[row][col].item<float>();
                std::cout << "    [" << row << ", " << col << "]: output=" << out_val << ", expected=" << exp_val
                          << std::endl;
            }
        } else {
            std::cout << "  PASS: Output matches expected" << std::endl;
        }

        EXPECT_TRUE(all_match_expected) << "DeepGemm output does NOT match expected output";

        if (all_match_expected) {
            std::cout << "\nSUCCESS: DeepGemm output matches expected!" << std::endl;
        } else {
            std::cout << "\nFAILURE: DeepGemm output does NOT match expected!" << std::endl;
        }
    }
};

TEST_F(DeepGemmDeterminismTest, TestDeterminismWithDumpedTensors) {
    std::string pt_dir = "/home/luoli.hn/work/good_cpp";
    RunDeepGemmDeterminismTestWithDumpedTensors(pt_dir, 10);
}