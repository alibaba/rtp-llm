#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/cpu_impl/CpuDevice.h"

using namespace std;

class CpuOpsTest: public DeviceTestBase {};

TEST_F(CpuOpsTest, testCopy) {
    vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto          A        = createHostBuffer({2, 3}, expected.data());
    // auto B = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::DEVICE}, {});
    // auto C = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::HOST}, {});
    // device_->copy({*A, *B});
    // device_->copy({*B, *C});
    // assertBufferValueEqual(*C, expected);
}

TEST_F(CpuOpsTest, testGemmOp) {}
