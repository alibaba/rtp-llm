#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/cpu_impl/CpuDevice.h"

using namespace std;
using namespace fastertransformer;

class CpuOpsTest: public DeviceTestBase<DeviceType::Cpu> {
public:
    void SetUp() override {
        DeviceTestBase<DeviceType::Cpu>::SetUp();
    }
    void TearDown() override {
        DeviceTestBase<DeviceType::Cpu>::TearDown();
    }
};

TEST_F(CpuOpsTest, testCopy) {
    vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto A = createHostBuffer({2, 3}, expected.data());
    // auto B = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::DEVICE}, {});
    // auto C = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::HOST}, {});
    // device_->copy({*A, *B});
    // device_->copy({*B, *C});
    // assertBufferValueEqual(*C, expected);
}

TEST_F(CpuOpsTest, testGemmOp) {
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
