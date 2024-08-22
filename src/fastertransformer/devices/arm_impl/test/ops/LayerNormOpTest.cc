#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class ArmLayerNormOpsTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        rtol_ = 1e-2;
        atol_ = 1e-2;
    }

protected:
    torch::Tensor rmsNorm(const torch::Tensor& input, const torch::Tensor& gamma, const torch::Tensor& beta) {
        return input * torch::rsqrt(torch::mean(input * input, -1, true) + 1e-6) * gamma + beta;
    }

    void testGeneralLayernorm(DataType data_type, uint16_t m, uint16_t n) {
        const auto torch_dtype     = dataTypeToTorchType(data_type);
        auto       input_tensor    = (torch::arange(m * n, m * n * 2) / (n * n)).reshape({m, n}).to(torch_dtype);
        auto       gamma_tensor    = (torch::ones({n})).to(torch_dtype);
        auto       beta_tensor     = (torch::zeros({n})).to(torch_dtype);
        auto       residual_tensor = torch::arange(m * n, -m * n, -2).reshape({m, n}).to(torch_dtype);

        auto      input   = tensorToBuffer(input_tensor, AllocationType::HOST);
        auto      gamma   = tensorToBuffer(gamma_tensor, AllocationType::HOST);
        auto      beta    = tensorToBuffer(beta_tensor, AllocationType::HOST);
        auto      weights = LayerNormWeights(gamma, beta);
        BufferPtr empty;
        auto      gamma_only_weights = LayerNormWeights(gamma, empty);
        auto      residual           = tensorToBuffer(residual_tensor, AllocationType::HOST);

        // test case 1: general layer norm without residual
        // pytorch 2.1.x ACL FP16 not enabled, segfault, comment out layernorm case
        // auto testcase1_output = device_->layernorm(LayernormParams(input,
        //                                                            nullptr,
        //                                                            weights,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            0.f,
        //                                                            1e-6,
        //                                                            false,
        //                                                            false,
        //                                                            NormType::layernorm));

        // auto expected_output = torch::layer_norm(input_tensor.to(torch::kFloat32),
        //                                          {n},
        //                                          gamma_tensor.to(torch::kFloat32),
        //                                          beta_tensor.to(torch::kFloat32),
        //                                          1e-6);
        // assertTensorClose(expected_output, bufferToTensor(*(testcase1_output.output)));

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

        auto expected_output = rmsNorm(
            input_tensor.to(torch::kFloat32), gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32));
        assertTensorClose(expected_output, bufferToTensor(*testcase2_output.output));
    }
};

TEST_F(ArmLayerNormOpsTest, testSimpleLayernorm) {
    const auto test_m = vector<uint16_t>({2});
    const auto test_n = vector<uint16_t>({2});
    for (const auto& m : test_m) {
        for (const auto& n : test_n) {
            testGeneralLayernorm(DataType::TYPE_FP16, m, n);
            testGeneralLayernorm(DataType::TYPE_FP32, m, n);
        }
    }
}