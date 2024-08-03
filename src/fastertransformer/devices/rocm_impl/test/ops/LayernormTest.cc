#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/alpha_layernorm_kernels.h"
#include "src/fastertransformer/devices/base_tests/LayerNormTest.hpp"

#include <torch/torch.h>
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/rocm/hip_utils.h"

using namespace std;
using namespace fastertransformer;

TEST_F(LayerNormTest, testAddBiasPerformance) {

// This test case is to verify performance diff of fused and seperate layernorm + add bias kernel

    cudaStream_t stream;
    hipStreamCreate(&stream);
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
    testAddBiasResidual();
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
