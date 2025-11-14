#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/arm_impl/test/ArmTestUtils.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/arm_impl/gemm_opt/ArmGemmKernel.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <torch/torch.h>

using namespace std;
using namespace rtp_llm;

class ArmGemmOpTest: public DeviceTestBase {
public:
    void BasicGemmOP(size_t m, size_t n, size_t k);
    void BasicGemmOP_FP16(size_t m, size_t n, size_t k);
    void BasicGemmOP_fp16fp16fp16(size_t m, size_t n, size_t k);
    void BasicGemmOP_fp32fp16fp16(size_t m, size_t n, size_t k);
    void BasicGemmOP_fp16fp16fp32(size_t m, size_t n, size_t k);
    void BasicGemmOP_fp32fp16fp32(size_t m, size_t n, size_t k);
    void BatchGemmOP(size_t b, size_t m, size_t n, size_t k);
    void TransposeBatchGemmOP(TransposeOperation op_a,
                              TransposeOperation op_b,
                              size_t             b,
                              size_t             m1,
                              size_t             k1,
                              size_t             k2,
                              size_t             n2,
                              size_t             m3,
                              size_t             n3);
};

void ArmGemmOpTest::BasicGemmOP_FP16(size_t m, size_t n, size_t k) {

    auto A_host = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = createDeviceBuffer<half>(A_host);
    auto B_device = createDeviceBuffer<half>(B_host);

    //    B_device = prepareGemmOptWeight(B_device);
    auto B_new = prepareGemmOptWeight(B_device);

    //    GemmParams params{*A_device, *B_device};
    GemmParams params{*A_device, *B_new};
    auto       C_device = device_->gemm(params);

    auto C_host = torch::matmul(A_host, B_host).to(torch::kHalf);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void ArmGemmOpTest::BasicGemmOP(size_t m, size_t n, size_t k) {
    // auto A_host = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    // auto B_host = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto A_host = torch::randn({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::randn({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    // auto A_device = createHostBuffer<float>({m, k}, tensorToBuffer(A_host, AllocationType::HOST)->data());
    // auto B_device = createHostBuffer<float>({k, n}, tensorToBuffer(B_host, AllocationType::HOST)->data());
    A_host *= 0.01;
    B_host *= 0.01;

    auto A_device = createDeviceBuffer<float>(A_host);
    auto B_device = createDeviceBuffer<float>(B_host);
    auto B_packed = prepareGemmOptWeight(B_device);

    GemmParams params{*A_device, *B_packed};
    auto       C_device = device_->gemm(params);
    printBufferData(*C_device, "C_device after gemm");

    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    // auto A      = bufferToTensor(*A_device);
    // auto B      = bufferToTensor(*B_device);
    auto C = bufferToTensor(*C_device);

    // ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
    assertTensorClose(C, C_host, 0.1, 0.02);
}

void ArmGemmOpTest::BasicGemmOP_fp32fp16fp16(size_t m, size_t n, size_t k) {
    auto A_host = torch::randn({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::randn({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    A_host *= 0.01;
    B_host *= 0.01;

    auto A_device = createDeviceBuffer<float>(A_host);
    auto B_device = createDeviceBuffer<half>(B_host);
    printBufferData(*B_device, "B_device before pack");

    auto B_packed = prepareKaiWeightBf16(B_device);
    printBufferData(*B_packed, "B_device after pack");

    GemmParams params{*A_device, *B_packed, std::nullopt, nullptr, DataType::TYPE_FP16};
    auto       C_device = device_->gemm(params);
    printBufferData(*C_device, "C_device after gemm");

    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto C      = bufferToTensor(*C_device);

    assertTensorClose(C, C_host, 0.1, 0.02);
}

void ArmGemmOpTest::BasicGemmOP_fp32fp16fp32(size_t m, size_t n, size_t k) {
    auto A_host = torch::randn({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::randn({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    A_host *= 0.01;
    B_host *= 0.01;

    auto A_device = createDeviceBuffer<float>(A_host);
    auto B_device = createDeviceBuffer<half>(B_host);
    printBufferData(*B_device, "B_device before pack");

    auto B_packed = prepareKaiWeightBf16(B_device);
    printBufferData(*B_packed, "B_device after pack");

    GemmParams params{*A_device, *B_packed, std::nullopt, nullptr, DataType::TYPE_FP32};
    auto       C_device = device_->gemm(params);
    printBufferData(*C_device, "C_device after gemm");

    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto C      = bufferToTensor(*C_device);

    assertTensorClose(C, C_host, 0.1, 0.02);
}

void ArmGemmOpTest::BasicGemmOP_fp16fp16fp16(size_t m, size_t n, size_t k) {
    auto A_host = torch::randn({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::randn({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    A_host *= 0.01;
    B_host *= 0.01;

    auto A_device = createDeviceBuffer<half>(A_host);
    auto B_device = createDeviceBuffer<half>(B_host);
    printBufferData(*B_device, "B_device before pack");

    auto B_packed = prepareKaiWeightBf16(B_device);
    printBufferData(*B_packed, "B_device after pack");

    GemmParams params{*A_device, *B_packed, std::nullopt, nullptr, DataType::TYPE_FP16};
    auto       C_device = device_->gemm(params);
    printBufferData(*C_device, "C_device after gemm");

    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto C      = bufferToTensor(*C_device);

    assertTensorClose(C, C_host, 0.1, 0.02);
}

void ArmGemmOpTest::BasicGemmOP_fp16fp16fp32(size_t m, size_t n, size_t k) {
    auto A_host = torch::randn({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::randn({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    A_host *= 0.01;
    B_host *= 0.01;

    auto A_device = createDeviceBuffer<half>(A_host);
    auto B_device = createDeviceBuffer<half>(B_host);
    printBufferData(*B_device, "B_device before pack");

    auto B_packed = prepareKaiWeightBf16(B_device);
    printBufferData(*B_packed, "B_device after pack");

    GemmParams params{*A_device, *B_packed, std::nullopt, nullptr, DataType::TYPE_FP32};

    auto C_device = device_->gemm(params);
    printBufferData(*C_device, "C_device after gemm");

    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto C      = bufferToTensor(*C_device);

    assertTensorClose(C, C_host, 0.1, 0.02);
}

void ArmGemmOpTest::BatchGemmOP(size_t b, size_t m, size_t n, size_t k) {
    auto A_host = torch::rand({(int)b, (int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)b, (int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = createDeviceBuffer<half>(A_host);
    auto B_device = createDeviceBuffer<half>(B_host);

    // B_device = prepareGemmOptWeight(B_device);
    auto B_packed = prepareGemmOptWeight(B_device);

    // GemmParams params{*A_device, *B_device};
    GemmParams params{*A_device, *B_packed};
    auto       C_device = device_->gemm(params);

    auto C_host = torch::matmul(A_host, B_host).to(torch::kHalf);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void ArmGemmOpTest::TransposeBatchGemmOP(TransposeOperation op_a,
                                         TransposeOperation op_b,
                                         size_t             b,
                                         size_t             m1,
                                         size_t             k1,
                                         size_t             k2,
                                         size_t             n2,
                                         size_t             m3,
                                         size_t             n3) {
    auto A_host = torch::rand({(int)b, (int)m1, (int)k1}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)b, (int)k2, (int)n2}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = createDeviceBuffer<float>(A_host);
    auto B_device = createDeviceBuffer<float>(B_host);

    // B_device = prepareGemmOptWeight(B_device);
    auto B_packed = prepareGemmOptWeight(B_device);

    // GemmParams params{*A_device, *B_device, nullopt, nullptr, DataType::TYPE_INVALID, DataType::TYPE_INVALID, op_a,
    // op_b};
    GemmParams params{
        *A_device, *B_packed, nullopt, nullptr, DataType::TYPE_INVALID, DataType::TYPE_INVALID, op_a, op_b};
    auto C_device = device_->gemm(params);

    if (op_a == TransposeOperation::TRANSPOSE) {
        A_host = A_host.transpose(1, 2);
    }
    if (op_b == TransposeOperation::TRANSPOSE) {
        B_host = B_host.transpose(1, 2);
    }
    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

TEST_F(ArmGemmOpTest, BasicGemmOpTest) {
    HWKernelConfig hw_kernel_config;
    if (hw_kernel_config.arm_gemm_use_kai) {
        return;
    }
    BasicGemmOP(2, 1024, 2048);
    // BasicGemmOP_FP16(2, 1024, 4);
    BasicGemmOP(4, 1024, 2048);
    BasicGemmOP(8, 1024, 2048);
    BasicGemmOP(1024, 1024, 2048);
    BasicGemmOP(4096, 1024, 2048);
}

TEST_F(ArmGemmOpTest, BasicGemmOPKaiTest) {
    HWKernelConfig hw_kernel_config;
    if (!hw_kernel_config.arm_gemm_use_kai) {
        return;
    }
    BasicGemmOP_fp32fp16fp32(4, 6, 1024);
    BasicGemmOP_fp16fp16fp16(4, 6, 1024);
    BasicGemmOP_fp16fp16fp32(4, 6, 1024);
    BasicGemmOP_fp32fp16fp16(4, 6, 1024);
}

TEST_F(ArmGemmOpTest, BatchGemmOpTest) {
    // BatchGemmOP(1, 8, 16, 4);
    // BatchGemmOP(1, 8, 16, 8);
    // BatchGemmOP(2, 8, 16, 8);
    // BatchGemmOP(4, 8, 16, 8);
    // BatchGemmOP(8, 8, 8, 8);
}

TEST_F(ArmGemmOpTest, TransposeBatchGemmOpTest) {
    // auto   tran = TransposeOperation::TRANSPOSE;
    // auto   none = TransposeOperation::NONE;
    // size_t b    = 128;
    // size_t m    = 64;
    // size_t n    = 8;
    // size_t k    = 16;
    // TransposeBatchGemmOP(none, none, b, m, k, k, n, m, n);
    //     TransposeBatchGemmOP(none, tran, b, m, k, n, k, m, n);
    //     TransposeBatchGemmOP(tran, tran, b, k, m, n, k, m, n);
    //     TransposeBatchGemmOP(tran, none, b, k, m, k, n, m, n);
}
