#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

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

    void syncCudaAndCheckError() {
        cudaDeviceSynchronize();
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("CUDA runtime error: ") + (_cudaGetErrorEnum(result)));
        }
    }
};

TEST_F(CudaOpsTest, testCopy) {
    vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto A = createHostBuffer({2, 3}, expected.data());
    auto B = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::DEVICE}, {});
    auto C = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::HOST}, {});
    device_->copy({*A, *B});
    device_->copy({*B, *C});

    syncCudaAndCheckError();
    assertBufferValueEqual(*C, expected);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
