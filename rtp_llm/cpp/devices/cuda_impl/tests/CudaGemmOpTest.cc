#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/base_tests/GemmOpTest.hpp"

using namespace std;

namespace rtp_llm {

class CudaGemmOpTest: public GemmOpTest {

public:
};

TEST_F(CudaGemmOpTest, BasicGemmOpTest) {
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP32);
    BasicQGemmOpTest(64, 64, 64, DataType::TYPE_FP16);
    BasicQGemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
    BasicQGemmOpTest(2, 2048, 4096, DataType::TYPE_FP16);
    // 结果正确，但int8 gemm跟float gemm之间的精度差较大，应改为int gemm对比
    qInt8QInt82DGemmOpTest(64, 64, 64);
    qInt8QInt82DGemmOpTest(2, 2048, 2048);
    qInt8QInt82DGemmOpTest(2, 4096, 4096);
}

TEST_F(CudaGemmOpTest, TransposeGemmOpTest) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t m    = 5;
    size_t n    = 1024;
    size_t k    = 4096;
    TransposeGemmOpTest(none, none, m, k, k, n, DataType::TYPE_FP16);
    TransposeGemmOpTest(none, tran, m, k, n, k, DataType::TYPE_FP16);
    TransposeGemmOpTest(tran, tran, k, m, n, k, DataType::TYPE_FP16);
    TransposeGemmOpTest(tran, none, k, m, k, n, DataType::TYPE_FP16);
    TransposeGemmOpTest(none, none, m, k, k, n, DataType::TYPE_FP32);
    TransposeGemmOpTest(none, tran, m, k, n, k, DataType::TYPE_FP32);
    TransposeGemmOpTest(tran, tran, k, m, n, k, DataType::TYPE_FP32);
    TransposeGemmOpTest(tran, none, k, m, k, n, DataType::TYPE_FP32);
}

TEST_F(CudaGemmOpTest, BatchGemmOpTest) {
    BatchGemmOpTest(1, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(2, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(4, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(8, 8, 8, 8, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(1, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(2, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(4, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(8, 8, 8, 8, DataType::TYPE_FP32);
}

TEST_F(CudaGemmOpTest, TransposeBatchGemmOpTest) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t b    = 128;
    size_t m    = 64;
    size_t n    = 8;
    size_t k    = 16;
    BatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP16);
    BatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP16);
    BatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP16);
    BatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP16);
    BatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP32);
    BatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP32);
    BatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP32);
    BatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP32);
}

TEST_F(CudaGemmOpTest, TransposeBatchMixFloatGemmOP) {
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

}  // namespace rtp_llm