#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class CudaSoftmaxOpTest: public CudaDeviceTestBase {
public:
    template<typename input_t>
    void BasicSoftmaxTest(size_t b,
                          size_t head_num,
                          size_t q_len,
                          size_t k_len,
                          float scale);
    
    template<typename input_t, typename output_t>
    void MixFloatSoftmaxTest(size_t b,
                             size_t head_num,
                             size_t q_len,
                             size_t k_len,
                             float scale);

};

template<typename input_t>
void CudaSoftmaxOpTest::BasicSoftmaxTest(size_t b,
                                         size_t head_num,
                                         size_t q_len,
                                         size_t k_len,
                                         float scale) {

    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto input_host = torch::rand(
        {(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto mask_host = torch::zeros(
        {(int)b, (int)q_len, (int)k_len}, tensor_options);

    auto output_host = torch::zeros(
        {(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto input_device = CreateDeviceBuffer<input_t>(input_host);
    auto mask_device = CreateDeviceBuffer<input_t>(mask_host);

    auto output_device = device_->softmax({*input_device, *mask_device, scale});
    assert(output_device == nullptr);
    
    mask_host = mask_host.reshape({(int)b, 1, (int)q_len, (int)k_len});
    auto result_ref = torch::softmax((input_host + mask_host) * scale, -1);

    auto result = bufferToTensor(*input_device);

    assertTensorClose(result, result_ref.to(result.dtype()));
}


template<typename input_t, typename output_t>
void CudaSoftmaxOpTest::MixFloatSoftmaxTest(size_t b,
                                            size_t head_num,
                                            size_t q_len,
                                            size_t k_len,
                                            float scale) {

    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto input_host = torch::rand(
        {(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto mask_host = torch::zeros(
        {(int)b, (int)q_len, (int)k_len}, tensor_options);

    auto output_host = torch::zeros(
        {(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto input_device = CreateDeviceBuffer<input_t>(input_host);
    auto mask_device = CreateDeviceBuffer<input_t>(mask_host);

    BufferPtr output_device = nullptr;
    output_device = device_->softmax({*input_device,
                                      *mask_device,
                                      scale,
                                      getTensorType<output_t>()});

    // if constexpr (std::is_same<output_t, half>::value) {
        
    // } else if constexpr (std::is_same<output_t, float>::value) {
    //     output_device = device_->softmax({*input_device,
    //                                        *mask_device,
    //                                        scale,
    //                                        DataType::TYPE_FP32});
    // }

    assert(output_device != nullptr);
    
    mask_host = mask_host.reshape({(int)b, 1, (int)q_len, (int)k_len});
    auto result_ref = torch::softmax((input_host + mask_host) * scale, -1);

    auto result = bufferToTensor(*output_device);

    assertTensorClose(result, result_ref.to(result.dtype()));
}



TEST_F(CudaSoftmaxOpTest, SoftmaxOpTest) {
    MixFloatSoftmaxTest<float, half>(16, 32, 128, 128, 1.0f);
    MixFloatSoftmaxTest<float, half>(16, 32, 128, 128, 2.0f);
}

