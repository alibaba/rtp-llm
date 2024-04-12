#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class CudaOpsTest: public DeviceTestBase {
public:
};

TEST_F(CudaOpsTest, testCopy) {
    vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto A = createHostBuffer({2, 3}, expected.data());
    auto B = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::DEVICE}, {});
    auto C = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::HOST}, {});
    device_->copy({*B, *A});
    device_->copy({*C, *B});

    assertBufferValueEqual(*C, expected);
}

TEST_F(CudaOpsTest, testCopyWithSlicing) {
    using TestT = int32_t;

    vector<TestT> input = {1, 2, 3, 4, 5, 6, 7, 8};
    auto src = createHostBuffer({4, 2}, input.data());
    auto dst = createBuffer<TestT>({2, 2}, {0, 0, 0, 0});

    device_->copy({*dst, *src, 0, 1, 2});

    assertBufferValueEqual<TestT>(*dst, {3, 4, 5, 6});

    device_->copy({*dst, *src, 1, 3, 1});
    assertBufferValueEqual<TestT>(*dst, {3, 4, 7, 8});
}

TEST_F(CudaOpsTest, testTranspose) {
    auto input = createBuffer<int32_t>({4, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    });
    std::vector<int32_t> expected = {
        1, 4, 7, 10,
        2, 5, 8, 11,
        3, 6, 9, 12
    };
    auto output = device_->transpose({*input});
    EXPECT_EQ(output->shape(), std::vector<size_t>({3, 4}));
    assertBufferValueEqual(*output, expected);

    sync_check_cuda_error();
}

