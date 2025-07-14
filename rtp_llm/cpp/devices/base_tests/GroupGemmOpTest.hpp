#pragma once
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <torch/torch.h>

using namespace rtp_llm;

class GroupGemmOpTest: public DeviceTestBase {
public:
    struct GroupGemmOpTestInput {
        std::vector<torch::Tensor> As;
        std::vector<torch::Tensor> Bs;
        std::vector<torch::Tensor> Cs;
    };

    struct GroupGemmOpTestOutput {
        std::vector<torch::Tensor> Cs;
    };

    // Basic + bias
    GroupGemmOpTestInput prepareInput(std::vector<std::tuple<size_t, size_t>> Aconfig,
                                      std::vector<std::tuple<size_t, size_t>> Bconfig,
                                      DataType                                type,
                                      bool                                    bias = false) {
        auto dtype = dataTypeToTorchType(type);
        auto num   = Aconfig.size();
        // ASSERT_TRUE((Aconfig.size() == Bconfig.size()));
        std::vector<torch::Tensor> As(num);
        std::vector<torch::Tensor> Bs(num);
        std::vector<torch::Tensor> Cs(num);
        for (int i = 0; i < num; i++) {
            auto m = (int)std::get<0>(Aconfig[i]);
            auto n = (int)std::get<1>(Bconfig[i]);
            auto k = (int)std::get<1>(Aconfig[i]);
            As[i]  = (torch::rand({m, k}, torch::Device(torch::kCPU)).to(dtype));
            Bs[i]  = (torch::rand({k, n}, torch::Device(torch::kCPU)).to(dtype));
            if (bias) {
                Cs[i] = (torch::ones({m, n}, torch::Device(torch::kCPU)).to(dtype));
            } else {
                Cs[i] = (torch::zeros({m, n}, torch::Device(torch::kCPU)).to(dtype));
            }
        }
        return GroupGemmOpTestInput({As, Bs, Cs});
    }

    GroupGemmOpTestOutput deviceGroupGemmOpRun(GroupGemmOpTestInput& input, bool bias = false) {
        auto                   num = input.As.size();
        std::vector<BufferPtr> a_buffers(num);
        std::vector<BufferPtr> b_buffers(num);
        std::vector<BufferPtr> c_buffers(num);
        for (int i = 0; i < num; i++) {
            auto a_buffer_ptr = tensorToBuffer(input.As[i]);
            a_buffers[i]      = (std::move(a_buffer_ptr));
            auto b_buffer_ptr = tensorToBuffer(input.Bs[i]);
            b_buffers[i]      = (std::move(b_buffer_ptr));
            auto c_buffer_ptr = tensorToBuffer(input.Cs[i]);
            c_buffers[i]      = (std::move(c_buffer_ptr));
        }
        GroupedGemmParams params{
            a_buffers, b_buffers, (bias) ? (std::optional<std::vector<BufferPtr>>)c_buffers : std::nullopt};
        auto                       result = device_->groupedGemm(params);
        std::vector<torch::Tensor> output(num);
        for (int i = 0; i < num; i++) {
            output[i] = (bufferToTensor(*(result.output[i])));
        }
        return GroupGemmOpTestOutput({output});
    }

    GroupGemmOpTestOutput torchGroupGemmOpRun(GroupGemmOpTestInput& input) {
        auto                       As = input.As;
        auto                       Bs = input.Bs;
        auto                       Cs = input.Cs;
        std::vector<torch::Tensor> output(Cs.size());
        for (int i = 0; i < As.size(); i++) {
            output[i] = Cs[i] + torch::matmul(As[i].to(torch::kFloat), Bs[i].to(torch::kFloat));
        }
        return GroupGemmOpTestOutput({output});
    }

    void groupGemmOpTest(std::vector<std::tuple<size_t, size_t>> Aconfig,
                         std::vector<std::tuple<size_t, size_t>> Bconfig,
                         DataType                                type,
                         bool                                    bias = false,
                         double                                  rtol = 0,
                         double                                  atol = 0) {
        auto input      = prepareInput(Aconfig, Bconfig, type, bias);
        auto result     = deviceGroupGemmOpRun(input, bias);
        auto result_ref = torchGroupGemmOpRun(input);
        for (int i = 0; i < result.Cs.size(); i++) {
            assertTensorClose(result.Cs[i].to(result_ref.Cs[i].scalar_type()), result_ref.Cs[i], rtol, atol);
        }
    }
};
