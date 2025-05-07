#include "maga_transformer/cpp/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/devices/base_tests/ActOpTest.hpp"

using namespace std;
using namespace rtp_llm;

class CudaActOpTest: public ActOpTest {};

TEST_F(CudaActOpTest, testSiluOp) {
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
