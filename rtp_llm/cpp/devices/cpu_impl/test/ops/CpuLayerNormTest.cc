#include "rtp_llm/cpp/devices/cpu_impl/CpuDevice.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <torch/torch.h>

using namespace std;
using namespace rtp_llm;

class CpuLayerNormTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        rtol_ = 1e-2;
        atol_ = 1e-2;
    }

protected:
    torch::Tensor rmsNorm(const torch::Tensor& input, const torch::Tensor& gamma, const double eps) {
        return input * torch::rsqrt(torch::mean(input * input, -1, true) + eps) * gamma;
    }

    void testGeneralLayernorm(DataType data_type, NormType norm_type, uint16_t m, uint16_t n) {
        const auto torch_dtype     = dataTypeToTorchType(data_type);
        auto       input_tensor    = (torch::arange(m * n, m * n * 2) / (n * n)).reshape({m, n}).to(torch_dtype);
        auto       gamma_tensor    = (torch::ones({n}) / 2).to(torch_dtype);
        auto       beta_tensor     = (torch::ones({n}) / 3).to(torch_dtype);
        auto       residual_tensor = torch::arange(m * n, -m * n, -2).reshape({m, n}).to(torch_dtype);

        auto      input   = tensorToBuffer(input_tensor);
        auto      gamma   = tensorToBuffer(gamma_tensor);
        auto      beta    = tensorToBuffer(beta_tensor);
        auto      weights = LayerNormWeights(gamma, beta);
        BufferPtr empty;
        gamma                   = tensorToBuffer(gamma_tensor);
        auto gamma_only_weights = LayerNormWeights(gamma, empty);
        auto residual           = tensorToBuffer(residual_tensor);

        // test case 1: general layer norm without residual
        auto testcase1_output = device_->layernorm(LayernormParams(input,
                                                                   nullptr,
                                                                   weights,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::layernorm));

        auto expected_output = torch::layer_norm(input_tensor.to(torch::kFloat32),
                                                 {n},
                                                 gamma_tensor.to(torch::kFloat32),
                                                 beta_tensor.to(torch::kFloat32),
                                                 1e-6);
        assertTensorClose(expected_output, bufferToTensor(*(testcase1_output.output)));

        // test case 2: rms norm without residual
        auto testcase2_output = device_->layernorm(LayernormParams(input,
                                                                   nullptr,
                                                                   weights,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::rmsnorm));

        expected_output = rmsNorm(input_tensor.to(torch::kFloat32), gamma_tensor.to(torch::kFloat32), 1e-6);
        assertTensorClose(expected_output, bufferToTensor(*testcase2_output.output));
    }
};

TEST_F(CpuLayerNormTest, testSimpleContextAttention) {
    const auto test_m = vector<uint16_t>({1, 2, 4, 8, 10, 20});
    const auto test_n = vector<uint16_t>({128, 256, 1024});
    for (const auto& m : test_m) {
        for (const auto& n : test_n) {
            printf("testing m = %d, n = %d \n", m, n);
            // testGeneralLayernorm(DataType::TYPE_BF16, NormType::layernorm, m, n);
            testGeneralLayernorm(DataType::TYPE_FP32, NormType::layernorm, m, n);
        }
    }
}
