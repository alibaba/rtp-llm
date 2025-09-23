#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

using namespace std;
using namespace rtp_llm;

class RocmOpsTest: public DeviceTestBase {};

TEST_F(RocmOpsTest, DevicePropertie) {
    GTEST_LOG_(INFO) << "\n\n************************\n ROCm device\n************************";

    ROCmDevice*      rocmDev     = static_cast<ROCmDevice*>(device_);
    DeviceProperties devProp     = device_->getDeviceProperties();
    hipDeviceProp_t* rocmDevProp = rocmDev->getRocmDeviceProperties();

    printf("\nDevice Properties:\n");
    printf("\tdevice id = %d\n", devProp.id);
    printf("\tdevice type = %d\n", devProp.type);

    printf("\nROCm Device Properties:\n");
    printf("\trocm device name = %s\n", rocmDevProp->name);
    printf("\trocm device gcnArchName = %s\n", rocmDevProp->gcnArchName);
    printf("\trocm device compute capability major = %d\n", rocmDevProp->major);
    printf("\trocm device compute capability minor = %d\n", rocmDevProp->minor);
    printf("\trocm device clockRate = %.2f(GHz)\n", rocmDevProp->clockRate / 1024.0 / 1024.0);
    printf("\trocm device totalGlobalMem = %.2f(GB)\n", rocmDevProp->totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("\trocm device sharedMemPerBlock = %.2f(KB)\n", rocmDevProp->sharedMemPerBlock / 1024.0);
    printf("\trocm device warpSize = %d\n", rocmDevProp->warpSize);
    printf("\trocm device maxThreadsPerBlock = %d\n", rocmDevProp->maxThreadsPerBlock);
    printf("\n");
}

TEST_F(RocmOpsTest, MemoryCopy) {
    GTEST_LOG_(INFO) << "\n\n************************\n memcpy test\n************************";

    unsigned long len           = 4096;
    auto          torchDataType = dataTypeToTorchType(DataType::TYPE_FP32);

    torch::Tensor tensorA   = torch::rand({(long)len}, torch::Device(torch::kCPU)).to(torchDataType);
    BufferPtr     h_bufferA = tensorToBuffer(tensorA, AllocationType::HOST);
    BufferPtr     d_bufferB = createBuffer({len}, DataType::TYPE_FP32, AllocationType::DEVICE);
    BufferPtr     h_bufferC = createBuffer({len}, DataType::TYPE_FP32, AllocationType::HOST);

    device_->copy({*d_bufferB, *h_bufferA});  // copy host -> device
    device_->copy({*h_bufferC, *d_bufferB});  // copy device -> host

    torch::Tensor tensorC = bufferToTensor(*h_bufferC);
    assertTensorClose(tensorA, tensorC);

    // TODO: calculate bandwidth
}

TEST_F(RocmOpsTest, testSelect) {
    auto src   = createBuffer<float>({6, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                              15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
    auto index = createBuffer<int32_t>({3}, {0, 2, 3});

    auto result   = device_->select({*src, *index});
    auto expected = torch::tensor({{0, 1, 2, 3, 4}, {10, 11, 12, 13, 14}, {15, 16, 17, 18, 19}}, torch::kFloat32);
    assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);

    auto src2    = device_->clone({*src, AllocationType::HOST});
    auto index2  = device_->clone({*index, AllocationType::HOST});
    auto result2 = device_->select({*src2, *index2});
    assertTensorClose(bufferToTensor(*result2), expected, 1e-6, 1e-6);
}

TEST_F(RocmOpsTest, testSelect1d) {
    auto src   = createBuffer<float>({2, 6}, {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15});
    auto index = createBuffer<int32_t>({3}, {0, 4, 5}, AllocationType::HOST);

    auto result   = device_->select({*src, *index, 1});
    auto expected = torch::tensor({{0, 4, 5}, {10, 14, 15}}, torch::kFloat32);
    assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);

    src      = createBuffer<float>({2, 5, 3}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
    index    = createBuffer<int32_t>({4}, {0, 1, 3, 4}, AllocationType::HOST);
    result   = device_->select({*src, *index, 1});
    expected = torch::tensor(
        {{0, 1, 2}, {3, 4, 5}, {9, 10, 11}, {12, 13, 14}, {15, 16, 17}, {18, 19, 20}, {24, 25, 26}, {27, 28, 29}},
        torch::kFloat32);
}
