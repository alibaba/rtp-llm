#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class CudaSoftmaxOpTest: public CudaDeviceTestBase {
public:
    void BasicSoftmaxTest(size_t b,
                          size_t head_num,
                          size_t q_len,
                          size_t k_len);

    void ScaleSoftmaxTest(float scale,
                          size_t b,
                          size_t head_num,
                          size_t q_len,
                          size_t k_len);

};

void CudaSoftmaxOpTest::BasicSoftmaxTest(size_t b,
                                         size_t head_num,
                                         size_t q_len,
                                         size_t k_len) {

    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto input_host = torch::rand(
        {(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto mask_host = torch::zeros(
        {(int)b, (int)q_len, (int)k_len}, tensor_options);

    auto output_host = torch::zeros(
        {(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto input_device = CreateDeviceBuffer<float>(input_host);
    auto mask_device = CreateDeviceBuffer<half>(mask_host);

    auto output_device = device_->softmax({*input_device, *mask_device});
    mask_host = mask_host.reshape({(int)b, 1, (int)q_len, (int)k_len});
    auto result_ref = torch::softmax(input_host + mask_host, -1);

    auto result = bufferToTensor(*output_device);

    assertTensorClose(result, result_ref);
}

void CudaSoftmaxOpTest::ScaleSoftmaxTest(float scale,
                                         size_t b,
                                         size_t head_num,
                                         size_t q_len,
                                         size_t k_len) {

    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto input_host = torch::rand(
        {(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto mask_host = torch::zeros(
        {(int)b, (int)q_len, (int)k_len}, tensor_options);

    auto output_host = torch::zeros(
        {(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);


    auto input_device = CreateDeviceBuffer<float>(input_host);
    auto mask_device = CreateDeviceBuffer<half>(mask_host);

    auto output_device = device_->softmax({*input_device, *mask_device, scale});
    mask_host = mask_host.reshape({(int)b, 1, (int)q_len, (int)k_len});
    auto result_ref = torch::softmax((input_host + mask_host) * scale, -1);

    auto result = bufferToTensor(*output_device);

    assertTensorClose(result, result_ref);
}



TEST_F(CudaSoftmaxOpTest, SoftmaxOpTest) {
    BasicSoftmaxTest(16, 32, 128, 128);
    ScaleSoftmaxTest(2.0f, 2, 2, 2, 2);
    ScaleSoftmaxTest(2.0f, 16, 32, 128, 128);
}



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
