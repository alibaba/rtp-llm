#include "src/fastertransformer/devices/rocm_impl/RocmTestUtils.h"
#include "src/fastertransformer/devices/base_tests/SoftmaxOpTest.hpp"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"


using namespace std;
using namespace fastertransformer;

class RocmSoftmaxOpTest: public SoftmaxOpTest {};

TEST_F(RocmSoftmaxOpTest, MixtureSoftmaxOpTest) {
    MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_FP32, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_FP32, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_FP16, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_FP16, DataType::TYPE_FP16);
    // MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_BF16, DataType::TYPE_BF16);
    // MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_BF16, DataType::TYPE_BF16);
}

