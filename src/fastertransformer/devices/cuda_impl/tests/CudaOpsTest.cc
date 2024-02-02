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
        allocator_ = device_->getAllocator();
        cpu_allocator_ = device_->getHostAllocator();
    }
    void TearDown() override {
        FtTestBase::TearDown();
    }

protected:
    DeviceBase* device_;
    IAllocator* allocator_;
    IAllocator* cpu_allocator_;
};

TEST_F(CudaOpsTest, testGemmOp) {
    Tensor A(allocator_, DataType::TYPE_FP16, {2, 4});
    Tensor B(allocator_, DataType::TYPE_FP16, {4, 3});
    Tensor C(allocator_, DataType::TYPE_FP16, {2, 3});
    Tensor workspace;

    GemmParams params {
        A, B, C,
        nullopt, nullopt,
        TransposeOperation::NONE, TransposeOperation::NONE,
        workspace
    };
    device_->gemm(params);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
