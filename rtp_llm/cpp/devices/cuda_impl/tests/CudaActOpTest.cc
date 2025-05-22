#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/base_tests/ActOpTest.hpp"

using namespace std;
using namespace rtp_llm;

class CudaActOpTest: public ActOpTest {};

TEST_F(CudaActOpTest, tescSiluOp) {
    BasicActOpTest(ActivationType::Silu, 100, 100, DataType::TYPE_FP16);
    BasicActOpTest(ActivationType::Silu, 1024, 1024, DataType::TYPE_FP16);
    BasicActOpTest(ActivationType::Silu, 1024, 4096, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Silu, 100, 100, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Silu, 1024, 1024, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Silu, 1024, 4096, DataType::TYPE_FP16);
}

TEST_F(CudaActOpTest, testGeluOp) {
    BasicActOpTest(ActivationType::Gelu, 100, 100, DataType::TYPE_FP16);
    BasicActOpTest(ActivationType::Gelu, 1024, 1024, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Gelu, 100, 100, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Gelu, 1024, 1024, DataType::TYPE_FP16);
}

TEST_F(CudaActOpTest, testFusedGateActOp) {
    // m must be divisible by 64
    // fp8有较大精度损失
    FuseGateActOpTest(ActivationType::Silu, 64, 128, DataType::TYPE_BF16);
}
