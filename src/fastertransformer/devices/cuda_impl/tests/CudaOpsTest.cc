#define private public
#include "src/fastertransformer/devices/testing/test_base.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

using namespace std;
using namespace fastertransformer;

class CudaOpsTest: public DeviceTestBase<DeviceType::Cuda> {
public:
    void SetUp() override {
        DeviceTestBase<DeviceType::Cuda>::SetUp();
    }
    void TearDown() override {
        DeviceTestBase<DeviceType::Cuda>::TearDown();
    }
};

TEST_F(CudaOpsTest, testCopy) {

}

TEST_F(CudaOpsTest, testGemmOp) {
    auto A = device_->allocateBuffer({DataType::TYPE_FP16, {2, 4}, AllocationType::DEVICE}, {});
    auto B = device_->allocateBuffer({DataType::TYPE_FP16, {4, 3}, AllocationType::DEVICE}, {});
    auto C = device_->allocateBuffer({DataType::TYPE_FP16, {2, 3}, AllocationType::DEVICE}, {});

    GemmParams params {A, B, C};
    device_->gemm(params);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
