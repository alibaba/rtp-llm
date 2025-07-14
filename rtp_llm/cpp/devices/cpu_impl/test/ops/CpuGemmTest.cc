#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/cpu_impl/CpuDevice.h"

// #include <torch/torch.h>

using namespace std;
using namespace rtp_llm;

class CpuGemmOpTest: public DeviceTestBase {
public:
    void GemmOp(DataType           data_type,
                size_t             m,
                size_t             n,
                size_t             k,
                TransposeOperation op_a = TransposeOperation::NONE,
                TransposeOperation op_b = TransposeOperation::NONE);

private:
    std::pair<BufferPtr, torch::Tensor> createBuffers(DataType data_type, size_t rows, size_t cols) {
        const auto torch_dtype = dataTypeToTorchType(data_type);
        auto       tensor      = torch::rand({(int)rows, (int)cols}, torch::Device(torch::kCPU)).to(torch_dtype);
        return {tensorToBuffer(tensor), tensor};
    }
};

void CpuGemmOpTest::GemmOp(
    DataType data_type, size_t m, size_t n, size_t k, TransposeOperation op_a, TransposeOperation op_b) {
    auto [A, A_tensor] =
        createBuffers(data_type, op_a == TransposeOperation::NONE ? m : k, op_a == TransposeOperation::NONE ? k : m);
    auto [B, B_tensor] =
        createBuffers(data_type, op_b == TransposeOperation::NONE ? k : n, op_b == TransposeOperation::NONE ? n : k);

    auto A_processed = (op_a == TransposeOperation::TRANSPOSE) ? A_tensor.to(torch::kFloat32).transpose(0, 1) :
                                                                 A_tensor.to(torch::kFloat32);
    auto B_processed = (op_b == TransposeOperation::TRANSPOSE) ? B_tensor.to(torch::kFloat32).transpose(0, 1) :
                                                                 B_tensor.to(torch::kFloat32);

    // Perform matrix multiplication
    auto expected_output = torch::matmul(A_processed, B_processed);

    GemmParams params{*A, *B, nullopt, nullptr, DataType::TYPE_INVALID, op_a, op_b};
    auto       test_result = device_->gemm(params);
    assertTensorClose(expected_output, bufferToTensor(*(test_result)), 1e-2, 1e-2);
}

TEST_F(CpuGemmOpTest, BasicGemmOpTestFP16) {
    size_t params[] = {2, 4, 8, 1024};
    for (size_t f : params) {
        GemmOp(DataType::TYPE_FP16, f, 1024, 2048, TransposeOperation::NONE, TransposeOperation::NONE);
        GemmOp(DataType::TYPE_FP16, f, 1024, 2048, TransposeOperation::TRANSPOSE, TransposeOperation::TRANSPOSE);
    }
}

TEST_F(CpuGemmOpTest, BasicGemmOpTestBF16) {
    size_t params[] = {2, 4, 8, 1024};
    for (size_t f : params) {
        GemmOp(DataType::TYPE_BF16, f, 1024, 2048, TransposeOperation::NONE, TransposeOperation::NONE);
        GemmOp(DataType::TYPE_BF16, f, 1024, 2048, TransposeOperation::TRANSPOSE, TransposeOperation::TRANSPOSE);
    }
}
