#include "rtp_llm/cpp/devices/base_tests/ActOpTest.hpp"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

using namespace std;
using namespace rtp_llm;

class RocmActOpTest: public ActOpTest {};

TEST_F(RocmActOpTest, testSiluOp) {
    BasicActOpTest(ActivationType::Silu, 100, 100, DataType::TYPE_FP16);
    BasicActOpTest(ActivationType::Silu, 1024, 1024, DataType::TYPE_FP16);
    BasicActOpTest(ActivationType::Silu, 1024, 4096, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Silu, 100, 100, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Silu, 1024, 1024, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Silu, 1024, 4096, DataType::TYPE_FP16);
}

TEST_F(RocmActOpTest, testGeluOp) {
    BasicActOpTest(ActivationType::Gelu, 100, 100, DataType::TYPE_FP16);
    BasicActOpTest(ActivationType::Gelu, 1024, 1024, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Gelu, 100, 100, DataType::TYPE_FP16);
    GateActOpTest(ActivationType::Gelu, 1024, 1024, DataType::TYPE_FP16);
}