#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/Weights.h"
#include "rtp_llm/cpp/kernels/rmsnormKernels.h"
#include "rtp_llm/cpp/kernels/rocm/layernorm_kernels.h"
#include "rtp_llm/cpp/kernels/add_residual_kernels.h"
#include "rtp_llm/cpp/devices/base_tests/LayerNormTest.hpp"

#include <torch/torch.h>
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"

using namespace std;
using namespace rtp_llm;

TEST_F(LayerNormTest, testFp16Conversion) {
    double        a      = 1.2345678;
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
    for (const auto& m : test_m) {
        for (const auto& n : test_n) {
            printf("testing m = %d, n = %d \n", m, n);
            testGeneralLayernorm(DataType::TYPE_FP16, NormType::layernorm, m, n);
            testGeneralLayernorm(DataType::TYPE_BF16, NormType::layernorm, m, n);
            testGeneralLayernorm(DataType::TYPE_FP32, NormType::layernorm, m, n);
        }
    }
}

TEST_F(LayerNormTest, testSimpleLayernormStride) {
    const auto test_m = vector<uint16_t>({1, 2, 4, 8, 10, 20});
    const auto test_n = vector<uint16_t>({128, 256, 1024});
    for (const auto& m : test_m) {
        for (const auto& n : test_n) {
            printf("testing m = %d, n = %d \n", m, n);
            testGeneralLayernormStride(DataType::TYPE_FP16, NormType::layernorm, m, n, n);
            testGeneralLayernormStride(DataType::TYPE_BF16, NormType::layernorm, m, n, n);
            testGeneralLayernormStride(DataType::TYPE_FP32, NormType::layernorm, m, n, n);
            testGeneralLayernormStride(DataType::TYPE_FP16, NormType::rmsnorm, m, n, n);
            testGeneralLayernormStride(DataType::TYPE_BF16, NormType::rmsnorm, m, n, n);
            testGeneralLayernormStride(DataType::TYPE_FP32, NormType::rmsnorm, m, n, n);
            testGeneralLayernormStride(DataType::TYPE_FP16, NormType::layernorm, m, n, n / 2);
            testGeneralLayernormStride(DataType::TYPE_BF16, NormType::layernorm, m, n, n / 2);
            testGeneralLayernormStride(DataType::TYPE_FP32, NormType::layernorm, m, n, n / 2);
            testGeneralLayernormStride(DataType::TYPE_FP16, NormType::rmsnorm, m, n, n / 2);
            testGeneralLayernormStride(DataType::TYPE_BF16, NormType::rmsnorm, m, n, n / 2);
            testGeneralLayernormStride(DataType::TYPE_FP32, NormType::rmsnorm, m, n, n / 2);
        }
    }
}
