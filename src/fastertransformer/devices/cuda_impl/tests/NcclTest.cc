#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"

#include <cuda.h>

using namespace std;
using namespace fastertransformer;

class NcclTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
#ifndef BUILD_MULTI_GPU
        abort();
#endif // !BUILD_MULTI_GPU
    }

};

void runTest(const size_t device_id) {
    cudaStream_t stream;
    check_cuda_error(cudaSetDevice(device_id));
    check_cuda_error(cudaStreamCreate(&stream));
}

TEST_F(NcclTest, testBasicComm) {
    size_t test_gpu_num = 2;
}

