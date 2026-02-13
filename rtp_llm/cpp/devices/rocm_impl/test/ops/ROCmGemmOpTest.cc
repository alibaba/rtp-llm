#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/base_tests/GemmOpTest.hpp"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

using namespace std;

namespace rtp_llm {
namespace rocm {

class ROCmGemmOpTest: public GemmOpTest {
public:
    GemmOpTestOutput RocmQ4x2GemmOpRun(GemmOpTestInput& input, int64_t group_size) {
        BufferPtr  A   = tensorToBuffer(input.A);
        BufferPtr  B   = tensorToBuffer(input.B);
        BufferPtr  Q4B = device_->quantize({*B, DataType::TYPE_QINT4X2, 1, group_size});
        BufferPtr  D   = device_->allocateBuffer({A->type(), {A->shape()[0], Q4B->shape()[1]}});
        GemmParams params{*A, *Q4B, std::nullopt, D};
        device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*D)});
    }

    void RocmQ4x2OpTest(long row, long col, int64_t group_size, DataType dtype) {
        ROCmDevice* rocDev = static_cast<ROCmDevice*>(device_);

        auto          ttype = dataTypeToTorchType(dtype);
        torch::Tensor tA    = torch::rand({row, col}).to(ttype);
        BufferPtr     A     = tensorToBuffer(tA);
        BufferPtr     Q4A   = rocDev->quantize({*A, DataType::TYPE_QINT4X2, 1, group_size});
        BufferPtr     DQ4A  = rocDev->dequantize({*Q4A, DataType::TYPE_QINT4X2, 1});
        torch::Tensor tDA   = bufferToTensor(*DQ4A);
        assertTensorClose(tA, tDA, 1, 1);
    }

    void RocmQ4x2GemmOpTest(size_t m, size_t n, size_t k, int64_t group_size, DataType dtype) {
        auto input      = PrepareGemmOpInput(m, n, k, dtype);
        auto result     = RocmQ4x2GemmOpRun(input, group_size);
        auto result_ref = BasicGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, 5e-1, 5e-1);
    }
};

TEST_F(ROCmGemmOpTest, Q4x2GemmOpTest) {
    RocmQ4x2OpTest(64, 64, 64, DataType::TYPE_FP16);
    RocmQ4x2OpTest(2048, 256, 64, DataType::TYPE_FP16);
    RocmQ4x2OpTest(1024, 2048, 128, DataType::TYPE_FP16);

#if USING_CK_INT4
    // Add ck int4 unit tests here.
#else
    RocmQ4x2GemmOpTest(64, 64, 64, 64, DataType::TYPE_FP16);
    RocmQ4x2GemmOpTest(256, 128, 256, 64, DataType::TYPE_FP16);
    RocmQ4x2GemmOpTest(1024, 2048, 512, 64, DataType::TYPE_FP16);

    RocmQ4x2GemmOpTest(128, 128, 128, 128, DataType::TYPE_FP16);
    RocmQ4x2GemmOpTest(256, 128, 256, 128, DataType::TYPE_FP16);
    RocmQ4x2GemmOpTest(1024, 2048, 512, 128, DataType::TYPE_FP16);
#endif
}

TEST_F(ROCmGemmOpTest, BasicGemmOpTest) {
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_BF16);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_BF16);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_BF16);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_BF16);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_BF16);
}

TEST_F(ROCmGemmOpTest, BasicFP8GemmOpTest) {
    BasicFP8GemmOpTest(1, 48, 16, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(2, 1024, 2048, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(8, 1024, 2048, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(1024, 1024, 2048, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(4096, 1024, 2048, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(1, 48, 16, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(1, 64, 16, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(1, 144, 48, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(1, 192, 48, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(1, 13056, 4352, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(1, 17408, 4352, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(3, 48, 16, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(3, 64, 16, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(3, 144, 48, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(3, 192, 48, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(3, 13056, 4352, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(3, 17408, 4352, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(48, 48, 16, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(48, 64, 16, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(48, 144, 48, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(48, 192, 48, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(48, 13056, 4352, DataType::TYPE_FP32);
    BasicFP8GemmOpTest(48, 17408, 4352, DataType::TYPE_FP32);
}

TEST_F(ROCmGemmOpTest, TransposeGemmOpTest) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t m    = 5;
    size_t n    = 1024;
    size_t k    = 4096;

    TransposeGemmOpTest(none, tran, m, k, n, k, DataType::TYPE_FP16);
    TransposeGemmOpTest(tran, tran, k, m, n, k, DataType::TYPE_FP16);
    TransposeGemmOpTest(tran, none, k, m, k, n, DataType::TYPE_FP16);
    TransposeGemmOpTest(none, none, m, k, k, n, DataType::TYPE_FP32);
    TransposeGemmOpTest(none, tran, m, k, n, k, DataType::TYPE_FP32);
    TransposeGemmOpTest(tran, tran, k, m, n, k, DataType::TYPE_FP32);
    TransposeGemmOpTest(tran, none, k, m, k, n, DataType::TYPE_FP32);
}

TEST_F(ROCmGemmOpTest, BatchGemmOpTest) {
    BatchGemmOpTest(1, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(2, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(4, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(8, 8, 8, 8, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(1, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(2, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(4, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(8, 8, 8, 8, DataType::TYPE_FP32);
}

TEST_F(ROCmGemmOpTest, TransposeBatchGemmOpTest) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t b    = 128;
    size_t m    = 64;
    size_t n    = 8;
    size_t k    = 16;
    BatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP32);
    BatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP32);
    BatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP32);
    BatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP32);
}

TEST_F(ROCmGemmOpTest, TransposeBatchMixFloatGemmOP) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t b    = 128;
    size_t m    = 64;
    size_t n    = 8;
    size_t k    = 16;
    MixtureBatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP32, DataType::TYPE_FP32);
}

}  // namespace rocm
}  // namespace rtp_llm