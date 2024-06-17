#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class CudaOpsTest: public DeviceTestBase {
public:
};

// TODO(wangyin.yx): transfer these tests to BasicDeviceTest when cpu ready.

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

TEST_F(CudaOpsTest, testConvert) {
    auto source = createBuffer<float>({7}, {0, -10, -1234, 1, 100, 10000, 3456});
    auto tensor = bufferToTensor(*source);
    auto testTypes = {
        DataType::TYPE_FP16, DataType::TYPE_BF16, DataType::TYPE_FP32,
        DataType::TYPE_INT32, DataType::TYPE_INT64};
    for (auto type1 : testTypes) {
        for (auto type2 : testTypes) {
            cout << "Testing " << type1 << " -> " << type2 << endl;
            auto src = tensorToBuffer(tensor.to(dataTypeToTorchType(type1)));
            auto output = device_->convert({src, type2});
            assertTensorClose(
                tensor.to(torch::kFloat32), bufferToTensor(*output).to(torch::kFloat32), 1, 1e-3);
            device_->syncAndCheck();
        }
    }
}

TEST_F(CudaOpsTest, testQBufferCopy) {
    auto tensor = torch::ones({5}, torch::kInt8);
    auto scales = torch::ones({5}, torch::kFloat);
    auto zeros  = torch::ones({5}, torch::kFloat);
    auto src = torchTensor2Buffer(tensor, scales, zeros);
    auto result_src = QBuffer2torchTensor(static_pointer_cast<const QBuffer>(src));
    EXPECT_TRUE(torch::equal(result_src[0], tensor));
    EXPECT_TRUE(torch::equal(result_src[1], scales));
    EXPECT_TRUE(torch::equal(result_src[2], zeros));
    auto dst_tensor = torch::zeros({5}, torch::kInt8);
    auto dst_scales = torch::zeros({5}, torch::kFloat);
    auto dst_zeros  = torch::zeros({5}, torch::kFloat);
    auto dst = torchTensor2Buffer(dst_tensor, dst_scales, dst_zeros);
    auto result_dst = QBuffer2torchTensor(static_pointer_cast<const QBuffer>(dst));
    EXPECT_TRUE(torch::equal(result_dst[0], dst_tensor));
    EXPECT_TRUE(torch::equal(result_dst[1], dst_scales));
    EXPECT_TRUE(torch::equal(result_dst[2], dst_zeros));
    device_->copy({*dst, *src});
    result_dst = QBuffer2torchTensor(static_pointer_cast<const QBuffer>(dst));
    EXPECT_TRUE(torch::equal(result_dst[0], tensor));
    EXPECT_TRUE(torch::equal(result_dst[1], scales));
    EXPECT_TRUE(torch::equal(result_dst[2], zeros));
}

TEST_F(CudaOpsTest, testConcat) {
    auto src1 = createBuffer<float>({4, 3}, {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11
    });
    auto src2 = createBuffer<float>({1, 3}, {
        111, 222, 333
    });
    auto src3 = createBuffer<float>({2, 3}, {
        1000, 1001, 1002,
        1003, 1004, 1005,
    });
    auto result = device_->concat({{src1, src2, src3}});
    device_->syncAndCheck();
    auto expected = torch::tensor({
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8},
        {9, 10, 11},
        {111, 222, 333},
        {1000, 1001, 1002},
        {1003, 1004, 1005}
    }, torch::kFloat32);
    assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);
}

TEST_F(CudaOpsTest, testSelect) {
    auto src = createBuffer<float>({6, 5}, {
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27, 28, 29
    });
    auto index = createBuffer<int32_t>({3}, {0, 2, 3});

    auto result = device_->select({*src, *index});
    auto expected = torch::tensor({
        {0, 1, 2, 3, 4},
        {10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19}
    }, torch::kFloat32);
    assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);

    auto src2 = device_->clone({*src, AllocationType::HOST});
    auto index2 = device_->clone({*index, AllocationType::HOST});
    auto result2 = device_->select({*src2, *index2});
    assertTensorClose(bufferToTensor(*result2), expected, 1e-6, 1e-6);
}
