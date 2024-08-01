#pragma once
#include "src/fastertransformer/devices/testing/TestBase.h"
#include <torch/torch.h>

using namespace fastertransformer;

class LoraLinearLayerTest: public DeviceTestBase {
public:

    struct LoraLinearLayerTestInput {
        torch::Tensor input;
        torch::Tensor weight;
        std::vector<int> input_lengths;
        std::vector<torch::Tensor> lora_a;
        std::vector<torch::Tensor> lora_b;
    };

    struct LoraLinearLayerTestOutput {
        torch::Tensor output;
    };

    LoraLinearLayerTestInput prepareInput(std::vector<int> input_lengths,
                                          int m,
                                          int n,
                                          int k,
                                          std::vector<int> ranks,
                                          DataType input_dtype,
                                          DataType lora_dtype)
    {
        auto input_dtype_ = dataTypeToTorchType(input_dtype);
        auto lora_dtype_  = dataTypeToTorchType(lora_dtype);
        auto batch_size = input_lengths.size();
        auto token_num = std::accumulate(input_lengths.begin(), input_lengths.end(), 0);

        auto input = torch::rand({token_num, k}, torch::Device(torch::kCPU)).to(input_dtype_);
        auto weight = 0.001 * torch::rand({k, n}, torch::Device(torch::kCPU)).to(input_dtype_);

        std::vector<torch::Tensor> lora_a(batch_size);
        std::vector<torch::Tensor> lora_b(batch_size);
        for (int i = 0; i < batch_size; i++) {
            lora_a[i] = 0.001 * torch::rand({k, ranks[i]}, torch::Device(torch::kCPU)).to(lora_dtype_);
            lora_b[i] = 0.001 * torch::rand({ranks[i], n}, torch::Device(torch::kCPU)).to(lora_dtype_);
        }
        return LoraLinearLayerTestInput({input, weight, input_lengths, lora_a, lora_b});
    }

    LoraLinearLayerTestOutput deviceLoraLinearLayerRun(LoraLinearLayerTestInput& params) {
        auto input = tensorToBuffer(params.input);
        auto weight = tensorToBuffer(params.weight);
        auto gemm_params = GemmParams(*input, *weight);
        auto lora_input_lengths = createHostBuffer<int32_t>({(size_t)params.input_lengths.size()}, params.input_lengths.data());
        std::vector<ConstBufferPtr> lora_as;
        std::vector<ConstBufferPtr> lora_bs;
        for (int i = 0; i < params.input_lengths.size(); i++) {
            lora_as.push_back(tensorToBuffer(params.lora_a[i]));
            lora_bs.push_back(tensorToBuffer(params.lora_b[i]));
        }
        auto lora_input = lora::LoraOpInput({lora_input_lengths, lora_as, lora_bs});
        auto output = device_->loraLinear(LoraLinearParams(gemm_params, lora_input));
        return LoraLinearLayerTestOutput({bufferToTensor(*output.output)});
    }

    LoraLinearLayerTestOutput deviceNoLoraLinearLayerRun(LoraLinearLayerTestInput& params) {
        auto input = tensorToBuffer(params.input);
        auto weight = tensorToBuffer(params.weight);
        auto gemm_params = GemmParams(*input, *weight);
        auto output = device_->loraLinear(LoraLinearParams(gemm_params));
        return LoraLinearLayerTestOutput({bufferToTensor(*output.output)});
    }


    LoraLinearLayerTestOutput torchLoraLinearLayerRun(LoraLinearLayerTestInput& input) {
        auto output = torch::matmul(input.input.to(torch::kFloat), input.weight.to(torch::kFloat));
        int start = 0;
        torch::Tensor lora_output = torch::empty(0);
        for (int i = 0; i < input.input_lengths.size(); i++) {
            auto intput_tmp = input.input.index({torch::indexing::Slice(start, start + input.input_lengths[i])}).contiguous();
            auto lora_output_tmp = torch::matmul(intput_tmp.to(torch::kFloat), input.lora_a[i].to(torch::kFloat));
            lora_output_tmp = torch::matmul(lora_output_tmp.to(torch::kFloat), input.lora_b[i].to(torch::kFloat));
            lora_output = torch::cat({lora_output, lora_output_tmp}, 0);
            start = start + input.input_lengths[i];
        }
        output = output + lora_output;
        return LoraLinearLayerTestOutput({output});
    }

    LoraLinearLayerTestOutput torchNoLoraLinearLayerRun(LoraLinearLayerTestInput& input) {
        auto output = torch::matmul(input.input.to(torch::kFloat), input.weight.to(torch::kFloat));
        return LoraLinearLayerTestOutput({output});
    }



    void loraLinearLayerTest(std::vector<int> input_lengths,
                             int m,
                             int n,
                             int k,
                             std::vector<int> ranks,
                             DataType input_dtype,
                             DataType lora_dtype)
    {
        auto input = prepareInput(input_lengths, m, n, k, ranks, input_dtype, lora_dtype);
        auto result = deviceLoraLinearLayerRun(input).output;
        auto ref = torchLoraLinearLayerRun(input).output;
        assertTensorClose(result.to(ref.scalar_type()), ref);
    }

    void noLoraLinearLayerTest(std::vector<int> input_lengths,
                             int m,
                             int n,
                             int k,
                             std::vector<int> ranks,
                             DataType input_dtype,
                             DataType lora_dtype)
    {
        auto input = prepareInput(input_lengths, m, n, k, ranks, input_dtype, lora_dtype);
        auto result = deviceNoLoraLinearLayerRun(input).output;
        auto ref = torchNoLoraLinearLayerRun(input).output;
        assertTensorClose(result.to(ref.scalar_type()), ref);

    }


};
