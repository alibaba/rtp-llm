#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include <torch/torch.h>

using namespace std;
using namespace rtp_llm;

class ArmSoftmaxOpTest: public DeviceTestBase {

public:
    template<typename input_t>
    void BasicSoftmaxTest(size_t b, size_t head_num, size_t q_len, size_t k_len, float scale);
};

template<typename input_t>
void ArmSoftmaxOpTest::BasicSoftmaxTest(size_t b, size_t head_num, size_t q_len, size_t k_len, float scale) {

    auto tensor_options = torch::TensorOptions(torch::kFloat).device(torch::Device(torch::kCPU));
    auto input_host     = torch::rand({(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto mask_host = torch::zeros({(int)b, (int)q_len, (int)k_len}, tensor_options);

    auto output_host = torch::zeros({(int)b, (int)head_num, (int)q_len, (int)k_len}, tensor_options);

    auto input_device = createDeviceBuffer<input_t>(input_host);
    auto mask_device  = createDeviceBuffer<input_t>(mask_host);

    auto output_device = device_->softmax({std::move(input_device), *mask_device, std::nullopt, scale});

    mask_host       = mask_host.reshape({(int)b, 1, (int)q_len, (int)k_len});
    auto result_ref = torch::softmax((input_host + mask_host) * scale, -1);

    auto result = bufferToTensor(*output_device);

    assertTensorClose(result, result_ref.to(result.dtype()));
}

TEST_F(ArmSoftmaxOpTest, SoftmaxOpTest) {
    BasicSoftmaxTest<float>(16, 32, 128, 128, 1.0f);
    BasicSoftmaxTest<float>(16, 32, 128, 128, 2.0f);
}
