#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/cuda/nccl/nccl_utils.h"

#include <nccl.h>
#include <cuda.h>
#include <future>

using namespace std;
using namespace rtp_llm;

class NcclTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
    }
};

struct NcclResponse {
    int64_t bcast_value;
};

void runTest(const size_t device_id, const size_t world_size, const ncclUniqueId& nccl_id) {
    cudaStream_t stream;
    check_cuda_value(cudaSetDevice(device_id));
    check_cuda_value(cudaStreamCreate(&stream));
    ncclComm_t nccl_comm;
    NCCLCHECK(ncclCommInitRank(&nccl_comm, world_size, nccl_id, device_id));

    float* cuda_data;
    size_t test_size = world_size * 3;
    cudaMalloc((void**)&cuda_data, sizeof(float*) * test_size);

    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclBcast(cuda_data + device_id, 1, ncclFloat, 0, nccl_comm, stream));
    NCCLCHECK(ncclGroupEnd());
    cudaFree(cuda_data);
    NCCLCHECK(ncclCommDestroy(nccl_comm));
}

TEST_F(NcclTest, testBasicComm) {
    // query available gpu num
    int device_count;
    check_cuda_value(cudaGetDeviceCount(&device_count));
    printf("cuda device_count: %d\n", device_count);
    if (device_count < 2) {
        return;
    }

    ncclUniqueId nccl_id;
    NCCLCHECK(ncclGetUniqueId(&nccl_id));
std:
    vector<future<void>> futures;
    for (size_t i = 0; i < device_count; i++) {
        auto future = async(launch::async, runTest, i, device_count, nccl_id);
        futures.push_back(move(future));
    }

    for (auto& future : futures) {
        future.get();
    }
}
