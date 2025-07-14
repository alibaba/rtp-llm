#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;
using namespace rtp_llm;

class BasicDeviceTest: public DeviceTestBase {};

TEST_F(BasicDeviceTest, testCopy) {
    vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto          A        = createHostBuffer({2, 3}, expected.data());
    auto          B        = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::DEVICE}, {});
    auto          C        = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::HOST}, {});
    device_->copy({*B, *A});
    device_->copy({*C, *B});

    assertBufferValueEqual(*C, expected);
}

TEST_F(BasicDeviceTest, testQueryStatus) {
    auto status = device_->getDeviceStatus();
    printf("device memory status: used_bytes=%zu, free_bytes=%zu, allocated_bytes=%zu, preserved_bytes=%zu\n",
           status.device_memory_status.used_bytes,
           status.device_memory_status.free_bytes,
           status.device_memory_status.allocated_bytes,
           status.device_memory_status.preserved_bytes);
}
