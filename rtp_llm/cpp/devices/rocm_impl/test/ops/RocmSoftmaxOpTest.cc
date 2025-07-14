#include "rtp_llm/cpp/devices/rocm_impl/RocmTestUtils.h"
#include "rtp_llm/cpp/devices/base_tests/SoftmaxOpTest.hpp"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

using namespace std;
using namespace rtp_llm;

class RocmSoftmaxOpTest: public SoftmaxOpTest {};

TEST_F(RocmSoftmaxOpTest, MixtureSoftmaxOpTest) {
    MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_FP32, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_FP32, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_FP16, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_FP16, DataType::TYPE_FP16);
    // MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_BF16, DataType::TYPE_BF16);
    // MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_BF16, DataType::TYPE_BF16);
}
