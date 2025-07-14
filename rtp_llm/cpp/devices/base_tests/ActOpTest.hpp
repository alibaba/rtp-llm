#pragma once
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include <torch/torch.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

using namespace rtp_llm;

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

    ActOpTestInput PrepareActOpInput(size_t m, size_t n, DataType dtype) {
        auto type      = dataTypeToTorchType(dtype);
        auto input     = torch::rand({(int)m, (int)n}, torch::Device(torch::kCPU)).to(type);
        auto gate      = torch::rand({(int)m, (int)n}, torch::Device(torch::kCPU)).to(type);
        auto gate_bias = torch::zeros({(int)m}, torch::Device(torch::kCPU)).to(type);
        return ActOpTestInput({input, gate, gate_bias});
    }

    ActOpTestOutput ActOpRun(ActOpTestInput& params, ActivationType atype) {
        auto input = tensorToBuffer(params.input);
        device_->activation({atype, input});
        auto output = bufferToTensor(*input);

        return ActOpTestOutput({output});
    }

    ActOpTestOutput GateActOpRun(ActOpTestInput& params, ActivationType atype) {
        auto input     = tensorToBuffer(params.input);
        auto gate      = tensorToBuffer(params.gate);
        auto gate_bias = tensorToBuffer(params.gate_bias);

        device_->activation({atype, input, std::nullopt, *gate, *gate_bias, std::nullopt});

        auto output = bufferToTensor(*input);

        return ActOpTestOutput({output});
    }

    ActOpTestOutput ActTorchRefRun(ActOpTestInput& params, ActivationType atype) {
        auto input = params.input;
        if (atype == ActivationType::Silu) {
            return ActOpTestOutput({torch::silu(input)});
        } else if (atype == ActivationType::Gelu) {
            return ActOpTestOutput({torch::gelu(input)});
        } else {
            throw std::runtime_error("invalid activation Type.");
        }
    }

    ActOpTestOutput GateActTorchRefRun(ActOpTestInput& params, ActivationType atype) {
        if (atype == ActivationType::Silu) {
            return ActOpTestOutput({torch::silu(params.gate) * params.input});
        } else if (atype == ActivationType::Gelu) {
            return ActOpTestOutput({torch::gelu(params.gate) * params.input});
        } else {
            throw std::runtime_error("invalid activation Type.");
        }
    }

    void BasicActOpTest(ActivationType atype, size_t m, size_t n, DataType dtype) {
        auto input      = PrepareActOpInput(m, n, dtype);
        auto result     = ActOpRun(input, atype);
        auto result_ref = ActTorchRefRun(input, atype);
        assertTensorClose(result.output.to(result_ref.output.type()), result_ref.output);
    }

    void GateActOpTest(ActivationType atype, size_t m, size_t n, DataType dtype) {
        auto input      = PrepareActOpInput(m, n, dtype);
        auto result     = GateActOpRun(input, atype);
        auto result_ref = GateActTorchRefRun(input, atype);
        assertTensorClose(result.output.to(result_ref.output.type()), result_ref.output);
    }

    void FuseGateActOpTest(ActivationType atype, size_t m, size_t n, DataType dtype) {
        auto input         = PrepareActOpInput(m, n, dtype);
        auto result_ref    = GateActTorchRefRun(input, atype);
        auto fused_input   = torch::cat({input.gate, input.input}, 1).contiguous();
        auto act_output    = device_->allocateBuffer({DataType::TYPE_BF16, {m, n}, AllocationType::DEVICE});
        auto out_          = device_->activation({atype,
                                                  tensorToBuffer(fused_input),
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  act_output,
                                                  true,
                                                  QScheme::Qfp8PerTokenBlock});
        auto output_kernel = bufferToTensor(reinterpret_cast<const QBuffer&>(*out_).kernel());
        auto output_scale  = bufferToTensor(reinterpret_cast<const QBuffer&>(*out_).scales()).transpose(0, 1);
        auto ref_output    = result_ref.output.view({(int)m, -1, 128});  //.to(torch::kFloat8_e4m3fn);
        auto ref_scale     = ref_output.abs().to(torch::kFloat32).amax(2).view({(int)m, (int)n / 128}).clamp(1e-4);
        auto scale         = 448.0 / ref_scale.unsqueeze(2);
        auto ref_quant     = ref_output.to(torch::kFloat32).mul(scale).to(torch::kFloat8_e4m3fn).view({(int)m, (int)n});
        ref_scale          = (1.0 / scale.to(torch::kFloat32)).view({(int)m, (int)n / 128});
        // assertTensorClose(ref_scale, output_scale);
        assertTensorClose(
            ref_quant.to(torch::kFloat32) * ref_scale, output_kernel.to(torch::kFloat32) * output_scale, 0.05, 0.05);
    }
};
