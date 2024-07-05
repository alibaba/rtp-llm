#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"

using namespace std;
using namespace fastertransformer;

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

    int  len           = 4096;
    auto torchDataType = dataTypeToTorchType(DataType::TYPE_FP32);

    torch::Tensor tensorA   = torch::rand({len}, torch::Device(torch::kCPU)).to(torchDataType);
    BufferPtr     h_bufferA = tensorToBuffer(tensorA, AllocationType::HOST);
    BufferPtr     d_bufferB = createBuffer({len}, DataType::TYPE_FP32, AllocationType::DEVICE);
    BufferPtr     h_bufferC = createBuffer({len}, DataType::TYPE_FP32, AllocationType::HOST);

    device_->copy({*d_bufferB, *h_bufferA});  // copy host -> device
    device_->copy({*h_bufferC, *d_bufferB});  // copy device -> host

    torch::Tensor tensorC = bufferToTensor(*h_bufferC);
    assertTensorClose(tensorA, tensorC);

    // TODO: calculate bandwidth
}

TEST_F(RocmOpsTest, TestOp) {
    GTEST_LOG_(INFO) << "\n\n************************\n TestOp(vector add)\n************************";

    int         len           = 64;
    ROCmDevice* rocmDev       = static_cast<ROCmDevice*>(device_);
    auto        torchDataType = dataTypeToTorchType(DataType::TYPE_FP32);

    torch::Tensor tensorA = torch::rand({1, len}, torch::Device(torch::kCPU)).to(torchDataType);
    torch::Tensor tensorB = torch::rand({1, len}, torch::Device(torch::kCPU)).to(torchDataType);
    torch::print(tensorA);
    torch::print(tensorB);
    torch::Tensor tensorC_ref = torch::add(tensorA, tensorB);

    BufferPtr     d_bufferA    = tensorToBuffer(tensorA);
    BufferPtr     d_bufferB    = tensorToBuffer(tensorB);
    BufferPtr     d_bufferC    = rocmDev->testVecAdd(d_bufferA, d_bufferB);
    torch::Tensor tensorC_rslt = bufferToTensor(*d_bufferC);
    torch::print(tensorC_rslt);

    assertTensorClose(tensorC_rslt, tensorC_ref);
}
