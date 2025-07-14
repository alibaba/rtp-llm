#pragma once
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <torch/torch.h>

using namespace rtp_llm;

class SoftmaxOpTest: public DeviceTestBase {
public:
    struct SoftmaxOpTestInput {
        torch::Tensor input;
        torch::Tensor mask;
        float         scale;
    };

    struct SoftmaxOpTestOutput {
        torch::Tensor out;
    };

    SoftmaxOpTestInput PrepareSoftmaxOpInput(
        size_t b, size_t head_num, size_t q_len, size_t k_len, float scale, DataType in_type, DataType out_type) {
        auto in_dtype           = dataTypeToTorchType(in_type);
        auto out_dtype          = dataTypeToTorchType(out_type);
        auto in_tensor_options  = torch::TensorOptions(in_dtype).device(torch::Device(torch::kCPU));
        auto out_tensor_options = torch::TensorOptions(out_dtype).device(torch::Device(torch::kCPU));
        auto input              = torch::rand({(int)b, (int)head_num, (int)q_len, (int)k_len}, in_tensor_options);

        auto mask = torch::zeros({(int)b, (int)q_len, (int)k_len}, out_tensor_options);
        return SoftmaxOpTestInput({input, mask});
    }

    SoftmaxOpTestOutput SoftmaxOpRun(SoftmaxOpTestInput& params) {
        bool is_cpu     = (this->device_->getDeviceProperties().type == DeviceType::Cpu);
        auto alloc_type = is_cpu ? AllocationType::HOST : AllocationType::DEVICE;
        auto input      = tensorToBuffer(params.input, alloc_type);
        auto mask       = tensorToBuffer(params.mask, alloc_type);

        auto output = device_->softmax({std::move(input), *mask, std::nullopt, params.scale, mask->type()});

        return SoftmaxOpTestOutput({bufferToTensor(*output)});
    }

    SoftmaxOpTestOutput SoftmaxTorchRefRun(SoftmaxOpTestInput& params) {
        auto mask = params.mask.reshape({params.mask.sizes()[0], 1, params.mask.sizes()[1], params.mask.sizes()[2]});
        return SoftmaxOpTestOutput(
            {torch::softmax(((params.input + mask) * params.scale).to(torch::kFloat32), -1).to(params.input.type())});
    }

    void MixtureSofmaxTest(
        size_t b, size_t head_num, size_t q_len, size_t k_len, float scale, DataType in_type, DataType out_type) {
        auto input      = PrepareSoftmaxOpInput(b, head_num, q_len, k_len, scale, in_type, out_type);
        auto result     = SoftmaxOpRun(input);
        auto result_ref = SoftmaxTorchRefRun(input);
        assertTensorClose(result.out.to(result_ref.out.type()), result_ref.out);
    }
};
