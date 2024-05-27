#pragma once
#include "src/fastertransformer/devices/testing/TestBase.h"
#include <torch/torch.h>

class ActOpTest: public DeviceTestBase {
public:

    struct ActOpTestInput {
        torch::Tensor input;
        torch::Tensor gate;
        torch::Tensor gate_bias;
    };

    struct ActOpTestOutput {
        torch::Tensor output;
    };

    ActOpTestInput PrepareActOpInput(size_t m,
                                     size_t n,
                                     DataType dtype)
    {
        auto type = dataTypeToTorchType(dtype);
        auto input = torch::rand({(int)m, (int)n}, torch::Device(torch::kCPU)).to(type);
        auto gate = torch::rand({(int)m, (int)n}, torch::Device(torch::kCPU)).to(type);
        auto gate_bias = torch::zeros({(int)m}, torch::Device(torch::kCPU)).to(type);
        return ActOpTestInput({input, gate, gate_bias});
    }

    ActOpTestOutput ActOpRun(ActOpTestInput& params,
                             ActivationType atype)
    {
        auto input = tensorToBuffer(params.input);
        device_->activation({atype, *input});
        auto output = bufferToTensor(*input);

       return ActOpTestOutput({output});

    }

    ActOpTestOutput GateActOpRun(ActOpTestInput& params,
                                 ActivationType atype)
    {
        auto input = tensorToBuffer(params.input);
        auto gate = tensorToBuffer(params.gate);
        auto gate_bias = tensorToBuffer(params.gate_bias);

        device_->activation({atype,
                             *input,
                             std::nullopt,
                             *gate,
                             *gate_bias});
        
        auto output = bufferToTensor(*input);

       return ActOpTestOutput({output});

    }

    ActOpTestOutput ActTorchRefRun(ActOpTestInput& params,
                                   ActivationType atype)
    {   
        auto input = params.input;
        if (atype == ActivationType::Silu) {
            return ActOpTestOutput({torch::silu(input)});
        } else if (atype == ActivationType::Gelu) {
            return ActOpTestOutput({torch::gelu(input)});
        } else {
            std::runtime_error("invalid activation Type.");
        }

    }

    ActOpTestOutput GateActTorchRefRun(ActOpTestInput& params,
                                       ActivationType atype)
    {   
        if (atype == ActivationType::Silu) {
            return ActOpTestOutput({torch::silu(params.gate) * params.input});
        } else if (atype == ActivationType::Gelu) {
            return ActOpTestOutput({torch::gelu(params.gate) * params.input});
        } else {
            std::runtime_error("invalid activation Type.");
        }

    }

    void BasicActOpTest(ActivationType atype,
                        size_t m,
                        size_t n,
                        DataType dtype)
    {
        auto input = PrepareActOpInput(m, n, dtype);
        auto result = ActOpRun(input, atype);
        auto result_ref = ActTorchRefRun(input, atype);
        assertTensorClose(result.output.to(result_ref.output.type()), result_ref.output);
    }

    void GateActOpTest(ActivationType atype,
                       size_t m,
                       size_t n,
                       DataType dtype)
    {
        auto input = PrepareActOpInput(m, n, dtype);
        auto result = GateActOpRun(input, atype);
        auto result_ref = GateActTorchRefRun(input, atype);
        assertTensorClose(result.output.to(result_ref.output.type()), result_ref.output);

    }

};


