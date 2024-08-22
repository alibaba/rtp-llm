#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/arm_impl/test/ArmTestUtils.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/arm_impl/type_bf16/hie_bfloat16.hpp"
#include "src/fastertransformer/devices/utils/Timer.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class ArmGemmOptOpTest: public DeviceTestBase {
public:
    void BasicGemmOP(size_t m, size_t n, size_t k);
    void BasicGemmOP_FP16(size_t m, size_t n, size_t k);
    void BatchGemmOP(size_t b, size_t m, size_t n, size_t k);
    void BatchGemmFP16OP(size_t b, size_t m, size_t n, size_t k, bool check_result=true);
    void TransposeBatchGemmOP(TransposeOperation op_a,
                              TransposeOperation op_b,
                              size_t             b,
                              size_t             m1,
                              size_t             k1,
                              size_t             k2,
                              size_t             n2,
                              size_t             m3,
                              size_t             n3);
    TimerRecorder timer_recorder_ = TimerRecorder();
    Timer timer_ = Timer();
};

void ArmGemmOptOpTest::BasicGemmOP_FP16(size_t m, size_t n, size_t k) {

    auto A_host = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = createDeviceBuffer<half>(A_host);
    auto B_device = createDeviceBuffer<half>(B_host);

    GemmParams params{*A_device, *B_device};
    auto       C_device = device_->gemm(params);

    auto C_host = torch::matmul(A_host, B_host).to(torch::kHalf);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void ArmGemmOptOpTest::BasicGemmOP(size_t m, size_t n, size_t k) {
    auto A_host = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = createHostBuffer<float>({m, k}, tensorToBuffer(A_host, AllocationType::HOST)->data());
    auto B_device = createHostBuffer<float>({k, n}, tensorToBuffer(B_host, AllocationType::HOST)->data());

    GemmParams params{*A_device, *B_device};
    auto       C_device = device_->gemm(params);

    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void ArmGemmOptOpTest::BatchGemmOP(size_t b, size_t m, size_t n, size_t k) {
    auto A_host = torch::rand({(int)b, (int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat) - 0.5;
    auto B_host = torch::rand({(int)b, (int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat) - 0.5;

    auto A_device = createDeviceBuffer<float>(A_host);
    auto B_device = createDeviceBuffer<float>(B_host);

    GemmParams params{*A_device, *B_device};

    timer_.reset();
    auto       C_device = device_->gemm(params);
    timer_recorder_.record(std::string("BatchGemmFP16OP") + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer_.elapsed_nano());


    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    auto C_host_slice = C_host.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    auto C_slice = C.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});

    // no nan
    ASSERT_TRUE((~torch::any(torch::isnan(C))).item<bool>());
    
    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}


void ArmGemmOptOpTest::BatchGemmFP16OP(size_t b, size_t m, size_t n, size_t k, bool check_result) {
    auto A_host = torch::rand({(int)b, (int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kHalf) - 0.5;
    auto B_host = torch::rand({(int)b, (int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kHalf) - 0.5;

    auto A_device = createDeviceBuffer<float16_t>(A_host);
    auto B_device = createDeviceBuffer<float16_t>(B_host);

    GemmParams params{*A_device, *B_device, nullopt, nullptr, DataType::TYPE_FP16};

    timer_.reset();
    auto       C_device = device_->gemm(params);
    timer_recorder_.record(std::string("BatchGemmFP16OP") + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer_.elapsed_nano());
    if (check_result) {
        std::cout << "BatchGemmFP16OP" << ", m=" << m << ", n=" << n << ", k=" << k << std::endl;
    }

    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    if (!check_result) {
        return;
    }

    // no nan
    ASSERT_TRUE((~torch::any(torch::isnan(C))).item<bool>());

    auto C_host = torch::matmul(A_host, B_host).to(torch::kHalf);
    
    auto C_host_slice = C_host.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    auto C_slice = C.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});

    std::cout << "C(pytorch): \n" << C_host_slice << std::endl;
    std::cout << "C(rtp-llm): \n" << C_slice << std::endl;

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void ArmGemmOptOpTest::TransposeBatchGemmOP(TransposeOperation op_a,
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

    GemmParams params{*A_device, *B_device, nullopt, nullptr, DataType::TYPE_INVALID, op_a, op_b};
    auto       C_device = device_->gemm(params);

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

TEST_F(ArmGemmOptOpTest, BatchGemmFP16OpTest) {

    auto m_list = vector<int>{1, 14, 144, 256, 512, 2035};

    for (int i = 0; i < 1; i++) {
        for (auto m : m_list) {
            BatchGemmFP16OP(1, m, 2048, 2048, false);
            BatchGemmFP16OP(1, m, 5504, 2048, false);
            BatchGemmFP16OP(1, m, 5504, 2048, false);
            BatchGemmFP16OP(1, m, 2048, 5504, false);
            BatchGemmFP16OP(1, m, 6144, 2048, false);
        }
    }
    timer_recorder_.print();
#ifdef GEMM_DEBUG
    ArmCpuDevice::print_time();
#endif
}
