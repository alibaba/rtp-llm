#include "rtp_llm/cpp/devices/base_tests/ActOpTest.hpp"
#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"

using namespace std;
using namespace rtp_llm;

class ArmActOpTest: public ActOpTest {};

TEST_F(ArmActOpTest, testSiluOp) {
    // pytorch 2.1.x ACL FP16 not enabled, segfault, comment out FP16 case
    // BasicActOpTest(ActivationType::Silu, 100, 100, DataType::TYPE_FP16);
    // BasicActOpTest(ActivationType::Silu, 1024, 1024, DataType::TYPE_FP16);
    // BasicActOpTest(ActivationType::Silu, 1024, 4096, DataType::TYPE_FP16);
    // Gate bias not supported yet
    // GateActOpTest(ActivationType::Silu, 100, 100, DataType::TYPE_FP16);
    // GateActOpTest(ActivationType::Silu, 1024, 1024, DataType::TYPE_FP16);
    // GateActOpTest(ActivationType::Silu, 1024, 4096, DataType::TYPE_FP16);
}

TEST_F(ArmActOpTest, testGeluOp) {
    // BasicActOpTest(ActivationType::Gelu, 100, 100, DataType::TYPE_FP16);
    // BasicActOpTest(ActivationType::Gelu, 1024, 1024, DataType::TYPE_FP16);
    // Gate bias not supported yet
    // GateActOpTest(ActivationType::Gelu, 100, 100, DataType::TYPE_FP16);
    // GateActOpTest(ActivationType::Gelu, 1024, 1024, DataType::TYPE_FP16);
}