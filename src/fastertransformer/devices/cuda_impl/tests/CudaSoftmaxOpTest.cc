#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/base_tests/SoftmaxOpTest.hpp"

using namespace std;
using namespace fastertransformer;

class CudaSoftmaxOpTest: public SoftmaxOpTest {};

TEST_F(CudaSoftmaxOpTest, MixtureSoftmaxOpTest) {
    MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_FP32, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_FP32, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_FP16, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_FP16, DataType::TYPE_FP16);
    MixtureSofmaxTest(16, 32, 128, 128, 1.0f, DataType::TYPE_BF16, DataType::TYPE_BF16);
    MixtureSofmaxTest(16, 32, 128, 128, 2.0f, DataType::TYPE_BF16, DataType::TYPE_BF16);
}
