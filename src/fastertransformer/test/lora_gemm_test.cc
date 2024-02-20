#include "src/fastertransformer/layers/LoraGemm.h"
#include "src/fastertransformer/core/Tensor.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"

#include "src/fastertransformer/cuda/cublas/cublas.h"
#include <gtest/gtest-message.h>
#include <gtest/gtest-test-part.h>
#include <gtest/gtest.h>

using namespace fastertransformer;
class LoraGemmTest: public ::testing::Test {
public:
    using num_t = half;

private:
    cublasHandle_t   cublasHandle;
    cublasLtHandle_t cublasltHandle;
    cudaStream_t     stream;
    std::mutex       mu;
    cublasAlgoMap    map;
    void             checkSize(int b, int m, int k, int r, int n);
    void             checkAlmostEqual(num_t* a, num_t* b, int n, float percent = 0.005);

public:
    void SetUp();
    void TearDown();
    void singleApplyLoRATest(int b, int m, int k, int r, int n, num_t* expected_output, int* lora_ids, int lora_num);

    void ApplyNullInputLengthsLoRATest(
        int b, int m, int k, int r, int n, num_t* expected_output, int* lora_ids, int lora_num);

    void ApplyDifferentInputLengthsLoRATest(
        int b, int m, int k, int r, int n, num_t* expected_output, int* lora_ids, int lora_num);

    Allocator<AllocatorType::CUDA>* allocator      = nullptr;
    cublasMMWrapper*                cublas_wrapper = nullptr;
    LoraGemm<num_t>*              gemm_runner    = nullptr;
    static constexpr int            B              = 2;
    static constexpr int            M              = 3;
    static constexpr int            K              = 4;
    static constexpr int            R              = 2;
    static constexpr int            N              = 5;
    num_t                           h_w[B * K * N] = {
        0.208130, 0.929799, 0.723109, 0.742336, 0.526296, 0.243658, 0.584592, 0.033153, 0.138717, 0.242235,
        0.815469, 0.793161, 0.278252, 0.481959, 0.819780, 0.997067, 0.698441, 0.567546, 0.835243, 0.205599,
        0.593172, 0.112347, 0.153457, 0.241708, 0.726237, 0.701080, 0.203824, 0.651054, 0.774486, 0.436891,
        0.519091, 0.615852, 0.810188, 0.980097, 0.114688, 0.316765, 0.696505, 0.914275, 0.935104, 0.941178,
    };
    num_t h_lora_a[B * K * R] = {
        0.496257,
        0.768222,
        0.088477,
        0.132030,
        0.307423,
        0.634079,
        0.490093,
        0.896445,
        0.455628,
        0.632306,
        0.348893,
        0.401717,
        0.022326,
        0.168859,
        0.293888,
        0.518522,
    };
    num_t h_lora_b[B * R * N] = {
        0.697668, 0.800011, 0.161029, 0.282269, 0.681609, 0.915194, 0.397100, 0.874156, 0.419408, 0.552907,
        0.952738, 0.036165, 0.185231, 0.373417, 0.305100, 0.932000, 0.175910, 0.269834, 0.150680, 0.031720,
    };
    num_t h_input[B * M * K] = {
        0.599507, 0.065209, 0.545996, 0.187197, 0.034023, 0.944246, 0.880180, 0.001236,
        0.593586, 0.415770, 0.417719, 0.271122, 0.692278, 0.203848, 0.683296, 0.752854,
        0.857936, 0.686956, 0.005132, 0.175652, 0.749658, 0.604651, 0.109958, 0.212090,
    };
    num_t h_null_input[B * 1 * K] = {0.599507, 0.065209, 0.545996, 0.187197, 0.034023, 0.944246, 0.880180, 0.001236};
    num_t h_output[M * M * N]     = {
        0.772555, 1.159351, 0.693839, 0.873585, 0.817399, 0.956146, 1.282621, 0.301520, 0.581482, 0.968444,
        0.835812, 1.315653, 0.713117, 0.926091, 0.811296, 1.146724, 1.064500, 1.480865, 1.698900, 1.378753,
        1.048819, 0.361908, 0.743653, 0.908690, 1.089097, 0.992845, 0.422904, 0.791696, 0.955588, 1.020821,
    };
    num_t h_null_output[M * 1 * N] = {
        0.772555, 1.159351, 0.693839, 0.873585, 0.817399, 0.956146, 1.282621, 0.301520, 0.581482, 0.968444};
    int h_input_lengths[B]           = {3, 3};
    int h_null_input_lengths[B]      = {1, 1};
    int h_different_input_lengths[B] = {1, 5};
};
void LoraGemmTest::SetUp() {
    cublasCreate(&cublasHandle);
    cublasLtCreate(&cublasltHandle);
    cudaStreamCreate(&stream);
    cublasSetStream(cublasHandle, stream);
    allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);
    cublas_wrapper = new cublasMMWrapper(cublasHandle, cublasltHandle, stream, &map, &mu, allocator);
    cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    gemm_runner = new LoraGemm<num_t>(stream, allocator, cublas_wrapper);
}
void LoraGemmTest::TearDown() {
    delete gemm_runner;
    delete cublas_wrapper;
    delete allocator;
    cublasDestroy(cublasHandle);
    cublasLtDestroy(cublasltHandle);
    cudaStreamDestroy(stream);
}
void LoraGemmTest::checkSize(int b, int m, int k, int r, int n) {
    ASSERT_LE(b, B);
    ASSERT_LE(m, M);
    ASSERT_LE(k, K);
    ASSERT_LE(r, R);
    ASSERT_LE(n, N);
}
void LoraGemmTest::checkAlmostEqual(num_t* a, num_t* b, int n, float percent) {
    for (int i = 0; i < n; i++) {
        float l     = static_cast<float>(a[i]);
        float r     = static_cast<float>(b[i]);
        float error = fabs(l - r) / fabs(l);
        ASSERT_LT(error, percent);
    }
}
void LoraGemmTest::singleApplyLoRATest(
    int b, int m, int k, int r, int n, num_t* expected_output, int* lora_ids, int lora_num) {
    checkSize(b, m, k, r, n);
    // device
    num_t* input  = (num_t*)allocator->malloc(sizeof(num_t) * b * m * k);
    num_t* output = (num_t*)allocator->malloc(sizeof(num_t) * b * m * n);
    cudaDeviceSynchronize();
    cudaMemcpy(input, h_input, sizeof(num_t) * b * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(output, h_output, sizeof(num_t) * b * m * n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // set LoRA weights
    LoRAWeight<num_t> lora_weights;
    num_t**           lora_a_array = new num_t*[lora_num];
    num_t**           lora_b_array = new num_t*[lora_num];
    for (int i = 0; i < lora_num; i++) {
        num_t* lora_a = (num_t*)allocator->malloc(sizeof(num_t) * k * r);
        num_t* lora_b = (num_t*)allocator->malloc(sizeof(num_t) * r * n);
        cudaDeviceSynchronize();
        lora_a_array[i]   = lora_a;
        lora_b_array[i]   = lora_b;
        num_t* cur_lora_a = h_lora_a + i * k * r;
        num_t* cur_lora_b = h_lora_b + i * r * n;
        cudaMemcpy(lora_a, cur_lora_a, sizeof(num_t) * k * r, cudaMemcpyHostToDevice);
        cudaMemcpy(lora_b, cur_lora_b, sizeof(num_t) * r * n, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        lora_weights.setLoRAWeight(i, lora_a, lora_b, r);
    }
    num_t* final_output = new num_t[b * m * n];
    gemm_runner->applyLoRA(b * m, b, h_input_lengths, k, n, lora_ids, &lora_weights, input, output);
    cudaDeviceSynchronize();
    cudaMemcpy(final_output, output, sizeof(num_t) * b * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkAlmostEqual(expected_output, final_output, b * m * n);
    allocator->free((void**)&input);
    allocator->free((void**)&output);
    for (int i = 0; i < lora_num; i++) {
        allocator->free((void**)&lora_a_array[i]);
        allocator->free((void**)&lora_b_array[i]);
    }
    delete[] final_output;
    delete[] lora_a_array;
    delete[] lora_b_array;
}
TEST_F(LoraGemmTest, ApplyLoRASimpleTest) {
    constexpr int b                          = 1;
    constexpr int m                          = 3;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        2.065056,
        2.000078,
        1.643932,
        1.444821,
        1.744666,
        1.865219,
        1.861871,
        0.982029,
        0.984167,
        1.614318,
        2.181695,
        2.194349,
        1.699152,
        1.520675,
        1.778559,
    };
    int lora_ids[b] = {0};
    singleApplyLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}
TEST_F(LoraGemmTest, ApplyLoRASimpleBatchTest) {
    constexpr int b                          = 2;
    constexpr int m                          = 3;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        2.065056, 2.000078, 1.643932, 1.444821, 1.744666, 1.865219, 1.861871, 0.982029, 0.984167, 1.614318,
        2.181695, 2.194349, 1.699152, 1.520675, 1.778559, 2.695979, 1.267407, 1.872953, 2.086061, 1.601371,
        2.547362, 0.546729, 1.115691, 1.300654, 1.326147, 2.368942, 0.593953, 1.134172, 1.313494, 1.235979,
    };
    int lora_ids[b] = {0, 1};
    singleApplyLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}
TEST_F(LoraGemmTest, ApplyLoRAHasNoLoRABatchTest) {
    constexpr int b                          = 2;
    constexpr int m                          = 3;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        0.772555, 1.159351, 0.693839, 0.873585, 0.817399, 0.956146, 1.282621, 0.301520, 0.581482, 0.968444,
        0.835812, 1.315653, 0.713117, 0.926091, 0.811296, 2.695979, 1.267407, 1.872953, 2.086061, 1.601371,
        2.547362, 0.546729, 1.115691, 1.300654, 1.326147, 2.368942, 0.593953, 1.134172, 1.313494, 1.235979,
    };
    int lora_ids[b] = {-1, 1};
    singleApplyLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}
TEST_F(LoraGemmTest, ApplyLoRASameLoRABatchTest) {
    constexpr int b                          = 2;
    constexpr int m                          = 3;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        1.672604, 1.277066, 0.921522, 1.098759, 0.947041, 1.816548, 1.392588, 0.517557, 0.800652, 1.097241,
        2.018720, 1.466441, 1.009881, 1.228019, 0.989125, 2.695979, 1.267407, 1.872953, 2.086061, 1.601371,
        2.547362, 0.546729, 1.115691, 1.300654, 1.326147, 2.368942, 0.593953, 1.134172, 1.313494, 1.235979,
    };
    int lora_ids[b] = {1, 1};
    singleApplyLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}
TEST_F(LoraGemmTest, ApplyLoRASimpleWithoutLoRATest) {
    constexpr int b                          = 1;
    constexpr int m                          = 3;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        0.772555,
        1.159351,
        0.693839,
        0.873585,
        0.817399,
        0.956146,
        1.282621,
        0.301520,
        0.581482,
        0.968444,
        0.835812,
        1.315653,
        0.713117,
        0.926091,
        0.811296,
    };
    int lora_ids[b] = {-1};
    singleApplyLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}
TEST_F(LoraGemmTest, ApplyLoRAWithoutLoRABatchTest) {
    constexpr int b                          = 2;
    constexpr int m                          = 3;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        0.772555, 1.159351, 0.693839, 0.873585, 0.817399, 0.956146, 1.282621, 0.301520, 0.581482, 0.968444,
        0.835812, 1.315653, 0.713117, 0.926091, 0.811296, 1.146724, 1.064500, 1.480865, 1.698900, 1.378753,
        1.048819, 0.361908, 0.743653, 0.908690, 1.089097, 0.992845, 0.422904, 0.791696, 0.955588, 1.020821,
    };
    int lora_ids[b] = {-1, -1};
    singleApplyLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}

void LoraGemmTest::ApplyNullInputLengthsLoRATest(
    int b, int m, int k, int r, int n, num_t* expected_output, int* lora_ids, int lora_num) {
    ASSERT_LE(b, B);
    ASSERT_LE(m, 1);
    ASSERT_LE(k, K);
    ASSERT_LE(r, R);
    ASSERT_LE(n, N);
    // device
    num_t* input  = (num_t*)allocator->malloc(sizeof(num_t) * b * m * k);
    num_t* output = (num_t*)allocator->malloc(sizeof(num_t) * b * m * n);
    cudaDeviceSynchronize();
    cudaMemcpy(input, h_null_input, sizeof(num_t) * b * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(output, h_null_output, sizeof(num_t) * b * m * n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // set LoRA weights
    LoRAWeight<num_t> lora_weights;
    num_t**           lora_a_array = new num_t*[lora_num];
    num_t**           lora_b_array = new num_t*[lora_num];
    for (int i = 0; i < lora_num; i++) {
        num_t* lora_a = (num_t*)allocator->malloc(sizeof(num_t) * k * r);
        num_t* lora_b = (num_t*)allocator->malloc(sizeof(num_t) * r * n);
        cudaDeviceSynchronize();
        lora_a_array[i]   = lora_a;
        lora_b_array[i]   = lora_b;
        num_t* cur_lora_a = h_lora_a + i * k * r;
        num_t* cur_lora_b = h_lora_b + i * r * n;
        cudaMemcpy(lora_a, cur_lora_a, sizeof(num_t) * k * r, cudaMemcpyHostToDevice);
        cudaMemcpy(lora_b, cur_lora_b, sizeof(num_t) * r * n, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        lora_weights.setLoRAWeight(i, lora_a, lora_b, r);
    }
    num_t* final_output = new num_t[b * 2 * n];
    gemm_runner->applyLoRA(b * m, b, h_null_input_lengths, k, n, lora_ids, &lora_weights, input, output);
    cudaDeviceSynchronize();
    cudaMemcpy(final_output, output, sizeof(num_t) * b * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkAlmostEqual(expected_output, final_output, b * m * n);
    allocator->free((void**)&input);
    allocator->free((void**)&output);
    for (int i = 0; i < lora_num; i++) {
        allocator->free((void**)&lora_a_array[i]);
        allocator->free((void**)&lora_b_array[i]);
    }
    delete[] final_output;
    delete[] lora_a_array;
    delete[] lora_b_array;
}

TEST_F(LoraGemmTest, ApplyWithoutLoRAWithNoLengthBatchTest) {
    constexpr int b                          = 2;
    constexpr int m                          = 1;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        0.772555, 1.159351, 0.693839, 0.873585, 0.817399, 0.956146, 1.282621, 0.301520, 0.581482, 0.968444};
    int lora_ids[b] = {-1, -1};
    ApplyNullInputLengthsLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}

TEST_F(LoraGemmTest, ApplySameLoRAWithNoLengthBatchTest) {
    constexpr int b                          = 2;
    constexpr int m                          = 1;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        1.672604, 1.277066, 0.921522, 1.098759, 0.947041, 1.816548, 1.392588, 0.517557, 0.800652, 1.097241};
    int lora_ids[b] = {1, 1};
    ApplyNullInputLengthsLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}

void LoraGemmTest::ApplyDifferentInputLengthsLoRATest(
    int b, int m, int k, int r, int n, num_t* expected_output, int* lora_ids, int lora_num) {
    checkSize(b, m, k, r, n);
    // device
    num_t* input  = (num_t*)allocator->malloc(sizeof(num_t) * b * m * k);
    num_t* output = (num_t*)allocator->malloc(sizeof(num_t) * b * m * n);
    cudaDeviceSynchronize();
    cudaMemcpy(input, h_input, sizeof(num_t) * b * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(output, h_output, sizeof(num_t) * b * m * n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // set LoRA weights
    LoRAWeight<num_t> lora_weights;
    num_t**           lora_a_array = new num_t*[lora_num];
    num_t**           lora_b_array = new num_t*[lora_num];
    for (int i = 0; i < lora_num; i++) {
        num_t* lora_a = (num_t*)allocator->malloc(sizeof(num_t) * k * r);
        num_t* lora_b = (num_t*)allocator->malloc(sizeof(num_t) * r * n);
        cudaDeviceSynchronize();
        lora_a_array[i]   = lora_a;
        lora_b_array[i]   = lora_b;
        num_t* cur_lora_a = h_lora_a + i * k * r;
        num_t* cur_lora_b = h_lora_b + i * r * n;
        cudaMemcpy(lora_a, cur_lora_a, sizeof(num_t) * k * r, cudaMemcpyHostToDevice);
        cudaMemcpy(lora_b, cur_lora_b, sizeof(num_t) * r * n, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        lora_weights.setLoRAWeight(i, lora_a, lora_b, r);
    }
    num_t* final_output = new num_t[b * m * n];
    gemm_runner->applyLoRA(b * m, b, h_different_input_lengths, k, n, lora_ids, &lora_weights, input, output);
    cudaDeviceSynchronize();
    cudaMemcpy(final_output, output, sizeof(num_t) * b * m * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkAlmostEqual(expected_output, final_output, b * m * n);
    allocator->free((void**)&input);
    allocator->free((void**)&output);
    for (int i = 0; i < lora_num; i++) {
        allocator->free((void**)&lora_a_array[i]);
        allocator->free((void**)&lora_b_array[i]);
    }
    delete[] final_output;
    delete[] lora_a_array;
    delete[] lora_b_array;
}

TEST_F(LoraGemmTest, DifferentLengthBatchTest) {
    constexpr int b                          = 2;
    constexpr int m                          = 3;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        1.672604, 1.277066, 0.921522, 1.098759, 0.947041, 1.816548, 1.392588, 0.517557, 0.800652, 1.097241,
        2.018720, 1.466441, 1.009881, 1.228019, 0.989125, 2.695979, 1.267407, 1.872953, 2.086061, 1.601371,
        2.547362, 0.546729, 1.115691, 1.300654, 1.326147, 2.368942, 0.593953, 1.134172, 1.313494, 1.235979,
    };
    int lora_ids[b] = {1, 1};
    ApplyDifferentInputLengthsLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}

TEST_F(LoraGemmTest, DifferentLengthDifferentLoraBatchTest) {
    constexpr int b                          = 2;
    constexpr int m                          = 3;
    constexpr int k                          = 4;
    constexpr int r                          = 2;
    constexpr int n                          = 5;
    num_t         expected_output[b * m * n] = {
        2.065056, 2.000078, 1.643932, 1.444821, 1.744666, 1.816548, 1.392588, 0.517557, 0.800652, 1.097241,
        2.018720, 1.466441, 1.009881, 1.228019, 0.989125, 2.695979, 1.267407, 1.872953, 2.086061, 1.601371,
        2.547362, 0.546729, 1.115691, 1.300654, 1.326147, 2.368942, 0.593953, 1.134172, 1.313494, 1.235979,
    };
    int lora_ids[b] = {0, 1};
    ApplyDifferentInputLengthsLoRATest(b, m, k, r, n, expected_output, lora_ids, b);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}