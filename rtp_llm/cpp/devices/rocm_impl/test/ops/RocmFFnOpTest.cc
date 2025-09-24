#include "rtp_llm/cpp/devices/rocm_impl/RocmTestUtils.h"
#include "rtp_llm/cpp/devices/base_tests/FfnLayerTest.hpp"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/core/Dispatch.h"

using namespace rtp_llm;

class RocmFfnLayerTest: public FfnLayerTest {};

TEST_F(RocmFfnLayerTest, Gate_Fp16_FfnOpTest) {
    FfnOpTest(4, 16, 16, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 32, 32, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 4096, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(128, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1, 2, 4096, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
}

TEST_F(RocmFfnLayerTest, NoGate_Fp16_FfnOpTest) {
    FfnOpTest(4, 16, 16, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(4, 32, 32, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 4096, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(128, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1, 2, 4096, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
}

// TEST_F(MoELayerTest, GateFp8MoEOpTest) {
//     MoEOpTest(10, 7168, 256, 32, 0, 8, ActivationType::Swiglu, DataType::TYPE_BF16, QScheme::Qfp8PerTokenBlock, 2e-1,
//     1e2);
// }