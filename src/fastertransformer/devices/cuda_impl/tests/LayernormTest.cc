#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class LayerNormTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        rtol_ = 1e-2;
        atol_ = 1e-2;
    }

protected:
    torch::Tensor rmsNorm(const torch::Tensor& input,
                          const torch::Tensor& gamma, const torch::Tensor& beta)
    {
        return input * torch::rsqrt(torch::mean(input * input, -1, true) + 1e-6) * gamma + beta;
    }

    void testGeneralLayernorm(DataType data_type, NormType norm_type, uint16_t m, uint16_t n) {
        const auto torch_dtype = dataTypeToTorchType(data_type);
        auto input_tensor = (torch::arange(m * n, m * n * 2) / (n * n)).reshape({m, n}).to(torch_dtype);
        auto gamma_tensor = (torch::ones({n}) / 2).to(torch_dtype);
        auto beta_tensor = (torch::ones({n}) / 3).to(torch_dtype);
        auto residual_tensor = torch::arange(m * n, - m * n, -2).reshape({m, n}).to(torch_dtype);

        auto input = tensorToBuffer(input_tensor);
        auto gamma = tensorToBuffer(gamma_tensor);
        auto beta = tensorToBuffer(beta_tensor);
        auto weights = LayerNormWeights({move(gamma), move(beta)});
        auto gamma_only_weights = LayerNormWeights({tensorToBuffer(gamma_tensor), nullptr});
        auto residual = tensorToBuffer(residual_tensor);
        auto output = createBuffer({m, n}, data_type);

        // test case 1: general layer norm without residual
        device_->layernorm(LayernormParams(
            *input, *output, nullopt, NormType::layernorm, weights, 1e-6));

        auto expected_output = torch::layer_norm(
            input_tensor.to(torch::kFloat32), {n},
            gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32), 1e-6);
        assertTensorClose(expected_output, bufferToTensor(*output));

        // extra: test case without beta
        device_->layernorm(LayernormParams(
            *input, *output, nullopt, NormType::layernorm, gamma_only_weights, 1e-6));

        // test case 2: general layer norm with residual and add_bias output
        output = createBuffer({m, n}, data_type);
        auto add_bias_output = createBuffer({m, n}, data_type);
        device_->layernorm(LayernormParams(
            *input, *output, *add_bias_output, NormType::layernorm, weights, 1e-6, *residual));

        expected_output = torch::layer_norm(
            (input_tensor + residual_tensor).to(torch::kFloat32), {n},
            gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32), 1e-6);
        auto expected_add_bias_output = input_tensor + residual_tensor;
        assertTensorClose(expected_output, bufferToTensor(*output));
        assertTensorClose(expected_add_bias_output, bufferToTensor(*add_bias_output));

        // test case 3: rms norm without residual
        device_->layernorm(LayernormParams(
            *input, *output, nullopt, NormType::rmsnorm, weights, 1e-6));

        expected_output = rmsNorm(
            input_tensor.to(torch::kFloat32),
            gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32));
        assertTensorClose(expected_output, bufferToTensor(*output));

        // extra: test case without beta
        device_->layernorm(LayernormParams(
            *input, *output, nullopt, NormType::rmsnorm, gamma_only_weights, 1e-6));

        // test case 4: rms norm with residual and add_bias output
        add_bias_output = createBuffer({m, n}, data_type);
        device_->layernorm(LayernormParams(
            *input, *output, *add_bias_output, NormType::rmsnorm, weights, 1e-6, *residual));

        expected_output = rmsNorm(
            (input_tensor + residual_tensor).to(torch::kFloat32),
            gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32));
        assertTensorClose(expected_output, bufferToTensor(*output));
        assertTensorClose(expected_add_bias_output, bufferToTensor(*add_bias_output));
    }

};

TEST_F(LayerNormTest, testAddBiasPerformance) {

// This test case is to verify performance diff of fused and seperate layernorm + add bias kernel

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    using TestType = float;
    const auto repeat_time = 100;
    const auto hidden_sizes = vector<size_t>({1024, 2048, 4096, 8192});
    const auto batch_sizes = vector<size_t>({1, 2, 4, 8, 16, 32, 64, 128, 256});
    for (const auto& hidden_size: hidden_sizes) {
        for (const auto& batch_size: batch_sizes) {
            const auto input = createDeviceBuffer<TestType>({batch_size, hidden_size}, nullptr);
            const auto gamma = createDeviceBuffer<TestType>({hidden_size}, nullptr);
            const auto beta = createDeviceBuffer<TestType>({hidden_size}, nullptr);
            const auto residual = createDeviceBuffer<TestType>({batch_size, hidden_size}, nullptr);
            const auto bias = createDeviceBuffer<TestType>({batch_size, hidden_size}, nullptr);

            {
                const auto start = chrono::high_resolution_clock::now();
                for (size_t i = 0; i < repeat_time; i++) {
                    invokeGeneralAddBiasResidualLayerNorm<TestType>(
                        (TestType*)input->data(),
                        (TestType*)input->data(),
                        (TestType*)input->data(),
                        (TestType*)bias->data(),
                        (TestType*)residual->data(),
                        (TestType*)gamma->data(),
                        (TestType*)beta->data(),
                        1e-6,
                        batch_size,
                        hidden_size,
                        stream
                    );
                }
                sync_check_cuda_error();

                const auto end = chrono::high_resolution_clock::now();
                const auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
                cout << "hidden_size: " << hidden_size << ", batch_size: " << batch_size << ", duration: " << duration << "us" << endl;
            }
            {
                const auto start = chrono::high_resolution_clock::now();
                for (size_t i = 0; i < repeat_time; i++) {
                    invokeAddBiasResidual(
                        (TestType*)input->data(),
                        (const TestType*)input->data(),
                        (const TestType*)residual->data(),
                        (const TestType*)nullptr,
                        (const TestType*)bias->data(),
                        nullptr,
                        nullptr,
                        batch_size,
                        hidden_size,
                        stream
                    );
                    invokeGeneralLayerNorm(
                        (TestType*)input->data(),
                        (TestType*)input->data(),
                        (TestType*)gamma->data(),
                        (TestType*)beta->data(),
                        1e-6,
                        batch_size,
                        hidden_size,
                        stream
                    );
                }
                sync_check_cuda_error();

                const auto end = chrono::high_resolution_clock::now();
                const auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
                cout << "hidden_size: " << hidden_size << ", batch_size: " << batch_size << ", duration: " << duration << "us" << endl;
            }
        }
    }
}

TEST_F(LayerNormTest, testFp16Conversion) {
    double a = 1.2345678;
    __nv_bfloat16 a_bf16 = a;
    cout << "a_bf16 = " << (float)a_bf16 << endl;
    double b = a_bf16;
    cout << b << endl;
    ASSERT_NEAR(a, b, 1e-3);
    half a_fp16 = a;
    cout << "a_fp16 = " << (float)a_fp16 << endl;
    b = a_fp16;
    cout << b << endl;
    ASSERT_NEAR(a, b, 1e-3);
}

TEST_F(LayerNormTest, testAddBiasResidual) {
    auto input = createBuffer<float>({2, 3}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
    auto norm_output = createBuffer<float>({2, 3}, {0, 0, 0, 0, 0, 0});
    const auto bias = createBuffer<float>({3}, {1, 2, 3});
    const auto residual = createBuffer<float>({2, 3}, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06});

    device_->syncAndCheck();
    device_->layernorm(LayernormParams(
        *input, *norm_output, *norm_output, NormType::add_bias, nullopt, nullopt,
        *residual, nullopt, *bias));

    assertBufferValueEqual(*input, vector<float>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}));
    assertBufferValueEqual(*norm_output, vector<float>({1.11, 2.22, 3.33, 1.44, 2.55, 3.66}));

    device_->layernorm(LayernormParams(
        *input, *input, *input, NormType::add_bias, nullopt, nullopt,
        *residual, nullopt, *bias));

    assertBufferValueEqual(*input, vector<float>({1.11, 2.22, 3.33, 1.44, 2.55, 3.66}));
}

TEST_F(LayerNormTest, testSimpleLayernorm) {
    const auto test_m = vector<uint16_t>({1, 2, 4, 8, 10, 20});
    const auto test_n = vector<uint16_t>({128, 256, 1024});
    for (const auto& m: test_m) {
        for (const auto& n: test_n) {
            printf("testing m = %d, n = %d \n", m, n);
            testGeneralLayernorm(DataType::TYPE_FP16, NormType::layernorm, m, n);
            testGeneralLayernorm(DataType::TYPE_BF16, NormType::layernorm, m, n);
            testGeneralLayernorm(DataType::TYPE_FP32, NormType::layernorm, m, n);
        }
    }
}
