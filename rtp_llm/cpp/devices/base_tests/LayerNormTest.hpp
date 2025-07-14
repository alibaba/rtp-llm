#pragma once
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <torch/torch.h>

using namespace std;
using namespace rtp_llm;

class LayerNormTest: public DeviceTestBase {
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

    void testAddBiasResidual() {
        auto       input    = createBuffer<float>({2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
        const auto bias     = createBuffer<float>({3}, {1, 2, 3});
        const auto residual = createBuffer<float>({2, 3}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06});

        device_->syncAndCheck();
        auto norm_output = device_->layernorm(LayernormParams(
            input, nullptr, nullopt, *residual, nullopt, *bias, 1.f, 0.f, false, false, NormType::alphanorm));

        assertBufferValueEqual(*input, vector<float>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}));
        assertBufferValueEqual(*norm_output.output, vector<float>({1.11, 2.22, 3.33, 1.44, 2.55, 3.66}));
        norm_output = device_->layernorm(LayernormParams(
            input, nullptr, nullopt, *residual, nullopt, *bias, 1.f, 0.f, true, false, NormType::alphanorm));

        assertBufferValueEqual(*norm_output.output, vector<float>({1.11, 2.22, 3.33, 1.44, 2.55, 3.66}));
    }

    void testGeneralLayernorm(DataType data_type, NormType norm_type, uint16_t m, uint16_t n) {
        const auto torch_dtype     = dataTypeToTorchType(data_type);
        auto       input_tensor    = (torch::arange(m * n, m * n * 2) / (n * n)).reshape({m, n}).to(torch_dtype);
        auto       gamma_tensor    = (torch::ones({n}) / 2).to(torch_dtype);
        auto       beta_tensor     = (torch::zeros({n})).to(torch_dtype);
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

        // extra: test case without beta
        auto testcase_extra_output = device_->layernorm(LayernormParams(input,
                                                                        nullptr,
                                                                        gamma_only_weights,
                                                                        std::nullopt,
                                                                        std::nullopt,
                                                                        std::nullopt,
                                                                        0.f,
                                                                        1e-6,
                                                                        false,
                                                                        false,
                                                                        NormType::layernorm));

        // test case 2: general layer norm with residual and add_bias output
        auto add_bias_output  = createBuffer({m, n}, data_type);
        auto testcase2_output = device_->layernorm(LayernormParams(input,
                                                                   add_bias_output,
                                                                   weights,
                                                                   *residual,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::layernorm));

        expected_output               = torch::layer_norm((input_tensor + residual_tensor).to(torch::kFloat32),
                                                          {n},
                                            gamma_tensor.to(torch::kFloat32),
                                            beta_tensor.to(torch::kFloat32),
                                            1e-6);
        auto expected_add_bias_output = input_tensor + residual_tensor;
        assertTensorClose(expected_output, bufferToTensor(*testcase2_output.output));
        assertTensorClose(expected_add_bias_output, bufferToTensor(*testcase2_output.before_norm_output));

        // test case 3: rms norm without residual
        auto testcase3_output = device_->layernorm(LayernormParams(input,
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

        expected_output = rmsNorm(
            input_tensor.to(torch::kFloat32), gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32));
        assertTensorClose(expected_output, bufferToTensor(*testcase3_output.output));

        // extra: test case without beta
        auto testcase3_extra_output = device_->layernorm(LayernormParams(input,
                                                                         nullptr,
                                                                         gamma_only_weights,
                                                                         std::nullopt,
                                                                         std::nullopt,
                                                                         std::nullopt,
                                                                         0.f,
                                                                         1e-6,
                                                                         false,
                                                                         false,
                                                                         NormType::rmsnorm));

        // test case 4: rms norm with residual and add_bias output
        add_bias_output       = createBuffer({m, n}, data_type);
        auto testcase4_output = device_->layernorm(LayernormParams(input,
                                                                   add_bias_output,
                                                                   weights,
                                                                   *residual,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::rmsnorm));

        expected_output = rmsNorm((input_tensor + residual_tensor).to(torch::kFloat32),
                                  gamma_tensor.to(torch::kFloat32),
                                  beta_tensor.to(torch::kFloat32));
        assertTensorClose(expected_output, bufferToTensor(*testcase4_output.output));
        assertTensorClose(expected_add_bias_output, bufferToTensor(*testcase4_output.before_norm_output));

        // test case 5: general layer norm with quantize
        auto testcase5_output = device_->layernorm(LayernormParams(input,
                                                                   nullptr,
                                                                   weights,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::layernorm,
                                                                   QScheme::Qint8PerToken));

        auto scales     = bufferToTensor(std::dynamic_pointer_cast<QBuffer>(testcase5_output.output)->scales());
        auto zeros      = torch::zeros_like(scales);
        expected_output = torch::layer_norm(input_tensor.to(torch::kFloat32),
                                            {n},
                                            gamma_tensor.to(torch::kFloat32),
                                            beta_tensor.to(torch::kFloat32),
                                            1e-6);
    }

    void
    testGeneralLayernormStride(DataType data_type, NormType norm_type, uint16_t m, uint16_t n, uint16_t norm_size) {
        const auto torch_dtype     = dataTypeToTorchType(data_type);
        auto       hidden_states_1 = (torch::arange(m * n, m * n * 2) / (n * n * 9)).reshape({m, n}).to(torch_dtype);
        auto hidden_states_2 = (torch::arange(m * n * 2, m * n * 3) / (n * n * 9)).reshape({m, n}).to(torch_dtype);

        auto hidden_states_3 = (torch::arange(m * n * 3, m * n * 4) / (n * n * 9)).reshape({m, n}).to(torch_dtype);

        // 拼接三个张量（最终维度为 [batch_size, 3n]）
        auto input_tensor = torch::cat({hidden_states_1, hidden_states_2, hidden_states_3}, /*dim=*/1);

        // auto input_tensor = (torch::arange(m * n, m * n * 2) / (n * n)).reshape({m, n}).to(torch_dtype);

        auto gamma_tensor    = (torch::ones({norm_size}) / 2).to(torch_dtype);
        auto beta_tensor     = (torch::ones({norm_size}) / 3).to(torch_dtype);
        auto residual_tensor = torch::arange(m * n, -m * n, -2).reshape({m, n}).to(torch_dtype);

        auto      input   = tensorToBuffer(input_tensor);
        auto      gamma   = tensorToBuffer(gamma_tensor);
        auto      beta    = tensorToBuffer(beta_tensor);
        auto      weights = LayerNormWeights(gamma, beta);
        BufferPtr empty;
        gamma                   = tensorToBuffer(gamma_tensor);
        auto gamma_only_weights = LayerNormWeights(gamma, empty);
        auto residual           = tensorToBuffer(residual_tensor);

        auto expected_output_1 = norm_type == NormType::layernorm ?
                                     torch::layer_norm(hidden_states_1.reshape({-1, norm_size}).to(torch::kFloat32),
                                                       {norm_size},
                                                       gamma_tensor.to(torch::kFloat32),
                                                       beta_tensor.to(torch::kFloat32),
                                                       1e-6)
                                         .reshape({m, -1}) :
                                     rmsNorm(hidden_states_1.reshape({-1, norm_size}).to(torch::kFloat32),
                                             gamma_tensor.to(torch::kFloat32),
                                             beta_tensor.to(torch::kFloat32))
                                         .reshape({m, -1});

        auto expected_output_2 = norm_type == NormType::layernorm ?
                                     torch::layer_norm(hidden_states_2.reshape({-1, norm_size}).to(torch::kFloat32),
                                                       {norm_size},
                                                       gamma_tensor.to(torch::kFloat32),
                                                       beta_tensor.to(torch::kFloat32),
                                                       1e-6)
                                         .reshape({m, -1}) :
                                     rmsNorm(hidden_states_2.reshape({-1, norm_size}).to(torch::kFloat32),
                                             gamma_tensor.to(torch::kFloat32),
                                             beta_tensor.to(torch::kFloat32))
                                         .reshape({m, -1});

        {
            auto testcase1_output_stride0 =
                device_->layernormWithStride(LayernormWithStrideParams({input, weights, 1e-6, norm_type, 0, n}));
            auto actual_output = bufferToTensor(*(testcase1_output_stride0.output)).slice(1, 0, n);
            assertTensorClose(expected_output_1, actual_output);
        }
        {
            auto testcase1_output =
                device_->layernormWithStride(LayernormWithStrideParams({input, weights, 1e-6, norm_type, n, n}));
            auto actual_output = bufferToTensor(*(testcase1_output.output)).slice(1, n, n + n);
            assertTensorClose(expected_output_2, actual_output);
        }
    }
};
