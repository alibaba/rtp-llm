#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "tests/unittests/gtest_utils.h"

using namespace std;
using namespace fastertransformer;

class CudaOpsTest: public FtTestBase {
public:
    void SetUp() override {
        FtTestBase::SetUp();
        device_ = DeviceFactory::getDevice(DeviceType::Cuda);
    }
    void TearDown() override {
        FtTestBase::TearDown();
    }

protected:
    DeviceBase* device_;
};

TEST_F(CudaOpsTest, testGemmOp) {
    auto A = device_->allocateBuffer({DataType::TYPE_FP16, {2, 4}}, {});
    auto B = device_->allocateBuffer({DataType::TYPE_FP16, {4, 3}}, {});
    auto C = device_->allocateBuffer({DataType::TYPE_FP16, {2, 3}}, {});

    GemmParams params {A, B, C};
    device_->gemm(params);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
