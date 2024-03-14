#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

using namespace std;
using namespace fastertransformer;

class CudaGemmOpTest: public DeviceTestBase<DeviceType::Cuda> {
public:

    double rtol_;
    double atol_;

    void SetUp() override {
        DeviceTestBase<DeviceType::Cuda>::SetUp();
        rtol_ = 1e-03;
        atol_ = 1e-03;
    }
    void TearDown() override {
        DeviceTestBase<DeviceType::Cuda>::TearDown();
    }

    void BasicGemmOP(size_t m, size_t n, size_t k);
    void BatchGemmOP(size_t b, size_t m, size_t n, size_t k);

    void TransposeBatchGemmOP(TransposeOperation op_a,
                              TransposeOperation op_b,
                              size_t b, 
                              size_t m1, size_t k1,
                              size_t k2, size_t m2,
                              size_t m3, size_t n3);
};

void CudaGemmOpTest::BasicGemmOP(size_t m, size_t n, size_t k) {
    auto A_host = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto C_host = torch::zeros({(int)m, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = CreateDeviceBuffer<half>(A_host);
    auto B_device = CreateDeviceBuffer<half>(B_host);
    auto C_device = CreateDeviceBuffer<half>(C_host);
    
    GemmParams params {*A_device, *B_device, *C_device};
    device_->gemm(params);

    C_host = torch::matmul(A_host, B_host);
    auto A     = CreateTensor(*A_device);
    auto B     = CreateTensor(*B_device);
    auto C     = CreateTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void CudaGemmOpTest::BatchGemmOP(size_t b, size_t m, size_t n, size_t k) {
    auto A_host = torch::rand(
        {(int)b, (int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand(
        {(int)b, (int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto C_host = torch::zeros(
        {(int)b, (int)m, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = CreateDeviceBuffer<half>(A_host);
    auto B_device = CreateDeviceBuffer<half>(B_host);
    auto C_device = CreateDeviceBuffer<float>(C_host);
    
    GemmParams params {*A_device, *B_device, *C_device};
    device_->gemm(params);

    C_host = torch::matmul(A_host, B_host);
    auto A     = CreateTensor(*A_device);
    auto B     = CreateTensor(*B_device);
    auto C     = CreateTensor(*C_device);

    // std::cout << C << std::endl;
    // std::cout << C_host << std::endl;

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void CudaGemmOpTest::TransposeBatchGemmOP(TransposeOperation op_a,
                                          TransposeOperation op_b,
                                          size_t b, 
                                          size_t m1, size_t k1,
                                          size_t k2, size_t m2,
                                          size_t m3, size_t n3) {
    auto A_host = torch::rand(
        {(int)b, (int)m1, (int)k1}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand(
        {(int)b, (int)k2, (int)m2}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto C_host = torch::zeros(
        {(int)b, (int)m3, (int)n3}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = CreateDeviceBuffer<half>(A_host);
    auto B_device = CreateDeviceBuffer<half>(B_host);
    auto C_device = CreateDeviceBuffer<float>(C_host);
    
    GemmParams params {op_a, op_b, *A_device, *B_device, *C_device};
    device_->gemm(params);
    if (op_a == TransposeOperation::TRANSPOSE) {
        A_host = A_host.transpose(1, 2);
    }
    if (op_b == TransposeOperation::TRANSPOSE) {
        B_host = B_host.transpose(1, 2);
    }
    C_host = torch::matmul(A_host, B_host);
    auto A     = CreateTensor(*A_device);
    auto B     = CreateTensor(*B_device);
    auto C     = CreateTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}


TEST_F(CudaGemmOpTest, BasicGemmOpTest) {
    BasicGemmOP(4096, 1024, 2048);

}

TEST_F(CudaGemmOpTest, BatchGemmOpTest) {
    BatchGemmOP(4, 8, 16, 32);
    BatchGemmOP(8, 8, 8, 8);

}

TEST_F(CudaGemmOpTest, TransposeBatchGemmOpTest) {
    auto tran = TransposeOperation::TRANSPOSE;
    auto none = TransposeOperation::NONE;
    size_t b = 128;
    size_t m = 64;
    size_t n = 8;
    size_t k = 16;
    TransposeBatchGemmOP(none, none, b, m, k, k, n, m, n);
    TransposeBatchGemmOP(none, tran, b, m, k, n, k, m, n);
    // TransposeBatchGemmOP(tran, tran, b, k, m, n, k, m, n);
    // TransposeBatchGemmOP(tran, none, b, k, m, k, n, m, n);

}



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
