#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/core/ExecOps.h"

using namespace std;
using namespace rtp_llm;

class BasicDeviceTest: public DeviceTestBase {};

TEST_F(BasicDeviceTest, testCopy) {
    vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto          A        = torch::tensor(expected, torch::kFloat32).reshape({2, 3});
    auto          B        = A.to(torch::kCUDA);
    auto          C        = B.cpu();
    runtimeSyncAndCheck();

    auto C_ptr = C.contiguous().data_ptr<float>();
    for (size_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(C_ptr[i], expected[i]);
    }
}

TEST_F(BasicDeviceTest, testQueryStatus) {
    auto status = getGpuExecStatus();
    printf("device memory status: used_bytes=%zu, free_bytes=%zu, allocated_bytes=%zu, available_bytes=%zu\n",
           status.device_memory_status.used_bytes,
           status.device_memory_status.free_bytes,
           status.device_memory_status.allocated_bytes,
           status.device_memory_status.available_bytes);
}
